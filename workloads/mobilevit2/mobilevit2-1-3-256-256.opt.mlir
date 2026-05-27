module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_1.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x8x8xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x512x8x8xf32>) -> tensor<512x64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<512x64xf32>, tensor<f32>) -> tensor<512xf32>
    return %12 : tensor<512xf32>
  }
  func.func private @region_1.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<1x256x4x16xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_1.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<1x256x2x2x4x4xf32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x2x2x4x4xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @region_1.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<1x256x8x8xf32>) -> tensor<f32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
    return %24 : tensor<f32>
  }
  func.func private @region_1.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<1x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x4x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<1x192x4x64xf32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<1x192x2x2x8x8xf32>, %arg1: tensor<f32>) -> tensor<128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x192x2x2x8x8xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x192x2x2x8x8xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %7 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<1x192x16x16xf32>) -> tensor<128xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 192, 1, 1>}> : (tensor<1x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
    return %24 : tensor<128xf32>
  }
  func.func private @region_1.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x128x4x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %7 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<1x128x4x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x128x4x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %7 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1x128x4x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x128x4x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %7 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<1x128x4x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x128x4x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %7 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<1x128x4x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<1x128x2x2x16x16xf32>, %arg1: tensor<f32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x128x2x2x16x16xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x128x2x2x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %7 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<1x128x32x32xf32>) -> tensor<256xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
    return %21 : tensor<256xf32>
  }
  func.func private @region_1.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter(%arg0: tensor<1x256x8x8xf32>, %arg1: tensor<1x2x4x2x4x256xf32>) -> tensor<1x256x8x8xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x4x2x4x256xf32>) -> tensor<64x1x256x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<1x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1x256x1x1xf32>) -> tensor<1x256x8x8xf32>
    return %23 : tensor<1x256x8x8xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<1x256x8x8xf32>) -> tensor<1x256x2x2x4x4xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<1x2x4x2x4x256xf32>
    %24 = stablehlo.transpose %23, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x4x2x4x256xf32>) -> tensor<1x256x2x2x4x4xf32>
    return %24 : tensor<1x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.1(%arg0: tensor<1x192x16x16xf32>, %arg1: tensor<1x2x8x2x8x192xf32>) -> tensor<1x192x16x16xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x8x2x8x192xf32>) -> tensor<256x1x192x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<1x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1x192x1x1xf32>) -> tensor<1x192x16x16xf32>
    return %23 : tensor<1x192x16x16xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<1x192x16x16xf32>) -> tensor<1x192x2x2x8x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 192, 1, 1>}> : (tensor<1x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<1x2x8x2x8x192xf32>
    %24 = stablehlo.transpose %23, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x8x2x8x192xf32>) -> tensor<1x192x2x2x8x8xf32>
    return %24 : tensor<1x192x2x2x8x8xf32>
  }
  func.func private @region_1.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.2(%arg0: tensor<1x128x32x32xf32>, %arg1: tensor<1x2x16x2x16x128xf32>) -> tensor<1x128x32x32xf32> {
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
    %19 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x16x2x16x128xf32>) -> tensor<1024x1x128x1x1xf32>
    %20 = "stablehlo.scatter"(%arg0, %18, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %21 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x128x32x32xf32>
    return %20 : tensor<1x128x32x32xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<1x128x32x32xf32>) -> tensor<1x128x2x2x16x16xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<1x2x16x2x16x128xf32>
    %21 = stablehlo.transpose %20, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x16x2x16x128xf32>) -> tensor<1x128x2x2x16x16xf32>
    return %21 : tensor<1x128x2x2x16x16xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<256xf32>, %arg1: tensor<1x256x4x16xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x2x4x2x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x256x2x2x4x4xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x2x4x2x4x256xf32>
    return %12 : tensor<1x2x4x2x4x256xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x2x8x2x8x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x192x2x2x8x8xf32>
    %17 = stablehlo.transpose %16, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x2x8x2x8x192xf32>
    return %17 : tensor<1x2x8x2x8x192xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x4x256xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x2x16x2x16x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x128x4x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x4x256xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x128x4x256xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x128x2x2x16x16xf32>
    %17 = stablehlo.transpose %16, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x2x16x2x16x128xf32>
    return %17 : tensor<1x2x16x2x16x128xf32>
  }
  func.func private @fused_add(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x2x2x16x16xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x1x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
    %6 = stablehlo.subtract %arg2, %5 : tensor<1x128x2x2x16x16xf32>
    %7 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %8 = stablehlo.multiply %7, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %9 = stablehlo.add %8, %cst_0 : tensor<1xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
    %13 = stablehlo.multiply %6, %12 : tensor<1x128x2x2x16x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x1x131072xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
    %17 = stablehlo.multiply %14, %16 : tensor<1x1x131072xf32>
    %18 = stablehlo.add %1, %17 : tensor<1x1x131072xf32>
    return %18 : tensor<1x1x131072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x2x2x8x8xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x1x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
    %6 = stablehlo.subtract %arg2, %5 : tensor<1x192x2x2x8x8xf32>
    %7 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %8 = stablehlo.multiply %7, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %9 = stablehlo.add %8, %cst_0 : tensor<1xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
    %13 = stablehlo.multiply %6, %12 : tensor<1x192x2x2x8x8xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x1x49152xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %17 = stablehlo.multiply %14, %16 : tensor<1x1x49152xf32>
    %18 = stablehlo.add %1, %17 : tensor<1x1x49152xf32>
    return %18 : tensor<1x1x49152xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<1x256x2x2x4x4xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x1x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %2 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<1x256x2x2x4x4xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x2x2x4x4xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x1x16384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1x1x16384xf32>
    %13 = stablehlo.add %1, %12 : tensor<1x1x16384xf32>
    return %13 : tensor<1x1x16384xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x128x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x128x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x128x128xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x128x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
    %12 = stablehlo.negate %10 : tensor<1x128x128x128xf32>
    %13 = stablehlo.exponential %12 : tensor<1x128x128x128xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x128x128x128xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x128x128x128xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x128x128x128xf32>
    return %16 : tensor<1x128x128x128xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x128x128xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x128x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x128x128xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x128x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x128x128xf32>
    return %10 : tensor<1x64x128x128xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x128x128xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x128x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x128x128xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x128x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x128x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
    %12 = stablehlo.negate %10 : tensor<1x64x128x128xf32>
    %13 = stablehlo.exponential %12 : tensor<1x64x128x128xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x64x128x128xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x64x128x128xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x64x128x128xf32>
    return %16 : tensor<1x64x128x128xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x128x128xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x64x128x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x64x128x128xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x64x128x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x128x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
    %12 = stablehlo.negate %10 : tensor<1x64x128x128xf32>
    %13 = stablehlo.exponential %12 : tensor<1x64x128x128xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x64x128x128xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x64x128x128xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x64x128x128xf32>
    return %16 : tensor<1x64x128x128xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x64x64xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %12 = stablehlo.negate %10 : tensor<1x256x64x64xf32>
    %13 = stablehlo.exponential %12 : tensor<1x256x64x64xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x256x64x64xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x256x64x64xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x256x64x64xf32>
    return %16 : tensor<1x256x64x64xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x64x64xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %12 = stablehlo.negate %10 : tensor<1x256x64x64xf32>
    %13 = stablehlo.exponential %12 : tensor<1x256x64x64xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x256x64x64xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x256x64x64xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x256x64x64xf32>
    return %16 : tensor<1x256x64x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x64x64xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %12 = stablehlo.negate %10 : tensor<1x256x64x64xf32>
    %13 = stablehlo.exponential %12 : tensor<1x256x64x64xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x256x64x64xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x256x64x64xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x256x64x64xf32>
    return %16 : tensor<1x256x64x64xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x32x128x128xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>) -> tensor<1x32x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x32x128x128xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<32xf32>
    %4 = stablehlo.rsqrt %3 : tensor<32xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x32x128x128xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x32x128x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x32x128x128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x128x128xf32>
    %12 = stablehlo.negate %10 : tensor<1x32x128x128xf32>
    %13 = stablehlo.exponential %12 : tensor<1x32x128x128xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x32x128x128xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x32x128x128xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x32x128x128xf32>
    return %16 : tensor<1x32x128x128xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x64x64xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x64x64xf32>
    %12 = stablehlo.negate %10 : tensor<1x128x64x64xf32>
    %13 = stablehlo.exponential %12 : tensor<1x128x64x64xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x128x64x64xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x128x64x64xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x128x64x64xf32>
    return %16 : tensor<1x128x64x64xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x32x32xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x32x32xf32>
    %12 = stablehlo.negate %10 : tensor<1x512x32x32xf32>
    %13 = stablehlo.exponential %12 : tensor<1x512x32x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x512x32x32xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x512x32x32xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x512x32x32xf32>
    return %16 : tensor<1x512x32x32xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<1x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1x128x64x64xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x128x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg5, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x64x64xf32>
    %11 = stablehlo.add %arg0, %10 : tensor<1x128x64x64xf32>
    return %11 : tensor<1x128x64x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x64x64xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x128x64x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<128xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x128x64x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x128x64x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128x64x64xf32>
    return %10 : tensor<1x128x64x64xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
    %12 = stablehlo.negate %10 : tensor<1x256x32x32xf32>
    %13 = stablehlo.exponential %12 : tensor<1x256x32x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x256x32x32xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x256x32x32xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x256x32x32xf32>
    return %16 : tensor<1x256x32x32xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
    %12 = stablehlo.negate %10 : tensor<1x256x32x32xf32>
    %13 = stablehlo.exponential %12 : tensor<1x256x32x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x256x32x32xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x256x32x32xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x256x32x32xf32>
    return %16 : tensor<1x256x32x32xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<1x256x4x256xf32>) -> tensor<1x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<1x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x256x4x256xf32>
    return %5 : tensor<1x256x4x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<1x256x4x256xf32>) -> tensor<1x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<1x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x256x4x256xf32>
    return %5 : tensor<1x256x4x256xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x32x32xf32>
    return %10 : tensor<1x256x32x32xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<1x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x256x32x32xf32>
    return %10 : tensor<1x256x32x32xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x768x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<1x768x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x768x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x768x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x768x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x768x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<1x768x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<1x768x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x768x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x768x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x768x16x16xf32>
    return %16 : tensor<1x768x16x16xf32>
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
  func.func private @fused_computation.300(%arg0: tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x128x4x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<128x4x256xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4x256xf32>, tensor<f32>) -> tensor<128x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x128x4x256xf32>
    return %20 : tensor<1x128x4x256xf32>
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
  func.func private @fused_computation.301(%arg0: tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x128x4x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<128x4x256xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4x256xf32>, tensor<f32>) -> tensor<128x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x128x4x256xf32>
    return %20 : tensor<1x128x4x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x16x16xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<1x512x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<1x512x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x512x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x512x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x512x16x16xf32>
    return %16 : tensor<1x512x16x16xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x4x256xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x128x4x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x4x256xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x128x4x256xf32>
    return %15 : tensor<1x128x4x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x4x256xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x128x4x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x4x256xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x128x4x256xf32>
    return %15 : tensor<1x128x4x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x4x256xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x128x4x256xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x128x4x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x4x256xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x128x4x256xf32>
    return %15 : tensor<1x128x4x256xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x384x16x16xf32>
    return %10 : tensor<1x384x16x16xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x384x16x16xf32>
    return %10 : tensor<1x384x16x16xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<1x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x384x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x384x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<1x384x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<1x384x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x384x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x384x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x384x16x16xf32>
    return %16 : tensor<1x384x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x384x4x64xf32>
    return %5 : tensor<1x384x4x64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x384x4x64xf32>
    return %5 : tensor<1x384x4x64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x384x4x64xf32>
    return %5 : tensor<1x384x4x64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x384x4x64xf32>
    return %5 : tensor<1x384x4x64xf32>
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
  func.func private @fused_computation.355(%arg0: tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x192x4x64xf32>
    return %20 : tensor<1x192x4x64xf32>
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
  func.func private @fused_computation.356(%arg0: tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x192x4x64xf32>
    return %20 : tensor<1x192x4x64xf32>
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
  func.func private @fused_computation.357(%arg0: tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x192x4x64xf32>
    return %20 : tensor<1x192x4x64xf32>
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
  func.func private @fused_computation.358(%arg0: tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x192x4x64xf32>
    return %20 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x4x64xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> tensor<1x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x4x64xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %7 = stablehlo.multiply %6, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %8 = stablehlo.add %7, %cst_0 : tensor<1xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<1x192x4x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x192x4x64xf32>
    %15 = stablehlo.add %0, %14 : tensor<1x192x4x64xf32>
    return %15 : tensor<1x192x4x64xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x768x8x8xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<1x768x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x768x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x768x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x768x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x768x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<1x768x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<1x768x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x768x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x768x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x768x8x8xf32>
    return %16 : tensor<1x768x8x8xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<1x512x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<1x512x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x512x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<1x512x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<1x512x8x8xf32>
    return %16 : tensor<1x512x8x8xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x512x4x16xf32>
    return %5 : tensor<1x512x4x16xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x512x4x16xf32>
    return %5 : tensor<1x512x4x16xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1x512x4x16xf32>
    return %5 : tensor<1x512x4x16xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<1x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<1x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<1x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<1x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x512x8x8xf32>
    return %10 : tensor<1x512x8x8xf32>
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
  func.func private @fused_computation.423(%arg0: tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x256x4x16xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x256x4x16xf32>
    return %20 : tensor<1x256x4x16xf32>
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
  func.func private @fused_computation.424(%arg0: tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x256x4x16xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x256x4x16xf32>
    return %20 : tensor<1x256x4x16xf32>
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
  func.func private @fused_computation.425(%arg0: tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1x256x4x16xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
    %18 = stablehlo.reduce(%17 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
    %20 = stablehlo.multiply %2, %19 : tensor<1x256x4x16xf32>
    return %20 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    return %10 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    return %10 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    return %10 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    return %10 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %2 = stablehlo.multiply %1, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %3 = stablehlo.add %2, %cst_0 : tensor<1xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<1x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<1x256x4x16xf32>
    return %10 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract(%arg0: tensor<1x256x2x2x4x4xf32>, %arg1: tensor<f32>) -> tensor<1x256x2x2x4x4xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x2x2x4x4xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x2x2x4x4xf32>
    return %4 : tensor<1x256x2x2x4x4xf32>
  }
  func.func private @fused_subtract.1(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract.2(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract.3(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract.4(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract.5(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @fused_subtract.6(%arg0: tensor<1x256x4x16xf32>, %arg1: tensor<f32>) -> tensor<1x256x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst : tensor<1xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x256x4x16xf32>
    return %4 : tensor<1x256x4x16xf32>
  }
  func.func private @region_1.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<2x512xf32>, %arg1: tensor<512xf32>) -> tensor<2xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<512xf32>) -> tensor<1x512xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x512xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x512xf32>) -> tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<512xf32>) -> tensor<2x512xf32>
    %5 = stablehlo.multiply %4, %arg0 : tensor<2x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2x512xf32>, tensor<f32>) -> tensor<2xf32>
    return %6 : tensor<2xf32>
  }
  func.func private @fused_broadcast() -> tensor<1x192x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x16x16xf32>
    return %0 : tensor<1x192x16x16xf32>
  }
  func.func private @fused_broadcast.1() -> tensor<1x128x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x32x32xf32>
    return %0 : tensor<1x128x32x32xf32>
  }
  func.func private @fused_broadcast.2() -> tensor<1x256x8x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x8x8xf32>
    return %0 : tensor<1x256x8x8xf32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<256xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.65(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.66(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.67(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.68(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.69(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.70(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.71(%arg0: tensor<128xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_add_computation(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<1x3x256x256xf32>) -> tensor<1x2xf32> {
    %0 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x8x8xf32>
      mhlo.return %287 : tensor<1x256x8x8xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1x256x8x8xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x16x16xf32>
      mhlo.return %287 : tensor<1x192x16x16xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1x192x16x16xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x32x32xf32>
      mhlo.return %287 : tensor<1x128x32x32xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1x128x32x32xf32>
    %3 = stablehlo.custom_call @__cudnn$convForward(%arg244, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x256x256xf32>, tensor<32x3x3x3xf32>) -> tuple<tensor<1x32x128x128xf32>, tensor<0xui8>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tensor<1x32x128x128xf32>, tensor<0xui8>>) -> tensor<1x32x128x128xf32>
    %5 = "mhlo.fusion"(%arg2, %arg1, %4, %arg194, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<1x32x128x128xf32>, %arg248: tensor<32xf32>, %arg249: tensor<32xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x32x128x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<32xf32>
      %291 = stablehlo.rsqrt %290 : tensor<32xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x32x128x128xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x32x128x128xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x32x128x128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x128x128xf32>
      %299 = stablehlo.negate %297 : tensor<1x32x128x128xf32>
      %300 = stablehlo.exponential %299 : tensor<1x32x128x128xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x32x128x128xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x32x128x128xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x32x128x128xf32>
      mhlo.return %303 : tensor<1x32x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x32x128x128xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.custom_call @__cudnn$convForward(%5, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x32x128x128xf32>, tensor<64x32x1x1xf32>) -> tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>) -> tensor<1x64x128x128xf32>
    %8 = "mhlo.fusion"(%arg5, %arg4, %7, %arg196, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<1x64x128x128xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x64x128x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<64xf32>
      %291 = stablehlo.rsqrt %290 : tensor<64xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x64x128x128xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x64x128x128xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x64x128x128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
      %299 = stablehlo.negate %297 : tensor<1x64x128x128xf32>
      %300 = stablehlo.exponential %299 : tensor<1x64x128x128xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x64x128x128xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x64x128x128xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x64x128x128xf32>
      mhlo.return %303 : tensor<1x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %9 = stablehlo.custom_call @__cudnn$convForward(%8, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x128x128xf32>, tensor<64x1x3x3xf32>) -> tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>
    %10 = stablehlo.get_tuple_element %9[0] : (tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>) -> tensor<1x64x128x128xf32>
    %11 = "mhlo.fusion"(%arg8, %arg7, %10, %arg198, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<1x64x128x128xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x64x128x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<64xf32>
      %291 = stablehlo.rsqrt %290 : tensor<64xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x64x128x128xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x64x128x128xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x64x128x128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
      %299 = stablehlo.negate %297 : tensor<1x64x128x128xf32>
      %300 = stablehlo.exponential %299 : tensor<1x64x128x128xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x64x128x128xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x64x128x128xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x64x128x128xf32>
      mhlo.return %303 : tensor<1x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %12 = stablehlo.custom_call @__cudnn$convForward(%11, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x128x128xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>
    %13 = stablehlo.get_tuple_element %12[0] : (tuple<tensor<1x64x128x128xf32>, tensor<0xui8>>) -> tensor<1x64x128x128xf32>
    %14 = "mhlo.fusion"(%arg11, %arg10, %13, %arg200, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<1x64x128x128xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x64x128x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<64xf32>
      %291 = stablehlo.rsqrt %290 : tensor<64xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x64x128x128xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x64x128x128xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1x64x128x128xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x64x128x128xf32>
      mhlo.return %297 : tensor<1x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x128x128xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x128x128xf32>
    %15 = stablehlo.custom_call @__cudnn$convForward(%14, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x128x128xf32>, tensor<128x64x1x1xf32>) -> tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>) -> tensor<1x128x128x128xf32>
    %17 = "mhlo.fusion"(%arg14, %arg13, %16, %arg202, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x128x128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<128xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x128x128x128xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x128x128x128xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x128x128x128xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %299 = stablehlo.negate %297 : tensor<1x128x128x128xf32>
      %300 = stablehlo.exponential %299 : tensor<1x128x128x128xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x128x128x128xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x128x128x128xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x128x128x128xf32>
      mhlo.return %303 : tensor<1x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %18 = stablehlo.custom_call @__cudnn$convForward(%17, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>
    %19 = stablehlo.get_tuple_element %18[0] : (tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>) -> tensor<1x128x64x64xf32>
    %20 = "mhlo.fusion"(%arg17, %arg16, %19, %arg204, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x64x64xf32>, %arg248: tensor<128xf32>, %arg249: tensor<128xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x128x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x128x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x128x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x128x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x64x64xf32>
      %299 = stablehlo.negate %297 : tensor<1x128x64x64xf32>
      %300 = stablehlo.exponential %299 : tensor<1x128x64x64xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x128x64x64xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x128x64x64xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x128x64x64xf32>
      mhlo.return %303 : tensor<1x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %21 = stablehlo.custom_call @__cudnn$convForward(%20, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x64x64xf32>, tensor<128x128x1x1xf32>) -> tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>
    %22 = stablehlo.get_tuple_element %21[0] : (tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>) -> tensor<1x128x64x64xf32>
    %23 = "mhlo.fusion"(%arg20, %arg19, %22, %arg206, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x64x64xf32>, %arg248: tensor<128xf32>, %arg249: tensor<128xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x128x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x128x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x128x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x128x64x64xf32>
      mhlo.return %297 : tensor<1x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %24 = stablehlo.custom_call @__cudnn$convForward(%23, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>
    %25 = stablehlo.get_tuple_element %24[0] : (tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>) -> tensor<1x256x64x64xf32>
    %26 = "mhlo.fusion"(%arg23, %arg22, %25, %arg208, %arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x64x64xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %299 = stablehlo.negate %297 : tensor<1x256x64x64xf32>
      %300 = stablehlo.exponential %299 : tensor<1x256x64x64xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x256x64x64xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x256x64x64xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x256x64x64xf32>
      mhlo.return %303 : tensor<1x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %27 = stablehlo.custom_call @__cudnn$convForward(%26, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>
    %28 = stablehlo.get_tuple_element %27[0] : (tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>) -> tensor<1x256x64x64xf32>
    %29 = "mhlo.fusion"(%arg26, %arg25, %28, %arg210, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x64x64xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %299 = stablehlo.negate %297 : tensor<1x256x64x64xf32>
      %300 = stablehlo.exponential %299 : tensor<1x256x64x64xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x256x64x64xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x256x64x64xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x256x64x64xf32>
      mhlo.return %303 : tensor<1x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %30 = stablehlo.custom_call @__cudnn$convForward(%29, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>
    %31 = stablehlo.get_tuple_element %30[0] : (tuple<tensor<1x128x64x64xf32>, tensor<0xui8>>) -> tensor<1x128x64x64xf32>
    %32 = "mhlo.fusion"(%23, %arg29, %arg28, %31, %arg212, %arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x64x64xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1x128x64x64xf32>, %arg249: tensor<128xf32>, %arg250: tensor<128xf32>):
      %287 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %288 = stablehlo.subtract %arg248, %287 : tensor<1x128x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg250, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x128x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x128x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x128x64x64xf32>
      %298 = stablehlo.add %arg245, %297 : tensor<1x128x64x64xf32>
      mhlo.return %298 : tensor<1x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x64x64xf32>
    %33 = stablehlo.custom_call @__cudnn$convForward(%32, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>
    %34 = stablehlo.get_tuple_element %33[0] : (tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>) -> tensor<1x256x64x64xf32>
    %35 = "mhlo.fusion"(%arg32, %arg31, %34, %arg214, %arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x64x64xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x64x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x64x64xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x64x64xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %299 = stablehlo.negate %297 : tensor<1x256x64x64xf32>
      %300 = stablehlo.exponential %299 : tensor<1x256x64x64xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x256x64x64xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x256x64x64xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x256x64x64xf32>
      mhlo.return %303 : tensor<1x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %36 = stablehlo.custom_call @__cudnn$convForward(%35, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>
    %37 = stablehlo.get_tuple_element %36[0] : (tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>) -> tensor<1x256x32x32xf32>
    %38 = "mhlo.fusion"(%arg35, %arg34, %37, %arg216, %arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x32x32xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x32x32xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x32x32xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
      %299 = stablehlo.negate %297 : tensor<1x256x32x32xf32>
      %300 = stablehlo.exponential %299 : tensor<1x256x32x32xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x256x32x32xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x256x32x32xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x256x32x32xf32>
      mhlo.return %303 : tensor<1x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %39 = stablehlo.custom_call @__cudnn$convForward(%38, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x32x32xf32>, tensor<256x256x1x1xf32>) -> tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>
    %40 = stablehlo.get_tuple_element %39[0] : (tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>) -> tensor<1x256x32x32xf32>
    %41 = "mhlo.fusion"(%arg38, %arg37, %40, %arg218, %arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x32x32xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x32x32xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x32x32xf32>
      mhlo.return %297 : tensor<1x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %42 = stablehlo.custom_call @__cudnn$convForward(%41, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x32x32xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>
    %43 = stablehlo.get_tuple_element %42[0] : (tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>) -> tensor<1x256x32x32xf32>
    %44 = "mhlo.fusion"(%arg41, %arg40, %43, %arg220, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x32x32xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x32x32xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x32x32xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
      %299 = stablehlo.negate %297 : tensor<1x256x32x32xf32>
      %300 = stablehlo.exponential %299 : tensor<1x256x32x32xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x256x32x32xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x256x32x32xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x256x32x32xf32>
      mhlo.return %303 : tensor<1x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %45 = stablehlo.custom_call @__cudnn$convForward(%44, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x32x32xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<1x128x32x32xf32>, tensor<0xui8>>
    %46 = stablehlo.get_tuple_element %45[0] : (tuple<tensor<1x128x32x32xf32>, tensor<0xui8>>) -> tensor<1x128x32x32xf32>
    %47 = "mhlo.fusion"(%46) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x32x32xf32>):
      %287 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %288 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %289 = stablehlo.add %287, %288 : tensor<2x16xi32>
      %290 = mhlo.bitcast %289 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %292 = stablehlo.compare LT, %290, %291 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %293 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %294 = stablehlo.add %290, %293 : tensor<1x1x2x16xi32>
      %295 = stablehlo.select %292, %294, %290 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %298 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %299 = stablehlo.compare LT, %289, %298 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %300 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %301 = stablehlo.add %289, %300 : tensor<2x16xi32>
      %302 = stablehlo.select %299, %301, %289 : tensor<2x16xi1>, tensor<2x16xi32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %304 = stablehlo.concatenate %297, %303, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %306 = "stablehlo.gather"(%arg245, %305) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1x128x1x1xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<1x2x16x2x16x128xf32>
      %308 = stablehlo.transpose %307, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x16x2x16x128xf32>) -> tensor<1x128x2x2x16x16xf32>
      mhlo.return %308 : tensor<1x128x2x2x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x32x32xf32>) -> tensor<1x128x2x2x16x16xf32>
    %48 = "mhlo.fusion"(%46) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x32x32xf32>):
      %287 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %288 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %289 = stablehlo.add %287, %288 : tensor<2x16xi32>
      %290 = mhlo.bitcast %289 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %292 = stablehlo.compare LT, %290, %291 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %293 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %294 = stablehlo.add %290, %293 : tensor<1x1x2x16xi32>
      %295 = stablehlo.select %292, %294, %290 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %298 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %299 = stablehlo.compare LT, %289, %298 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %300 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %301 = stablehlo.add %289, %300 : tensor<2x16xi32>
      %302 = stablehlo.select %299, %301, %289 : tensor<2x16xi1>, tensor<2x16xi32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %304 = stablehlo.concatenate %297, %303, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %306 = "stablehlo.gather"(%arg245, %305) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1x128x1x1xf32>
      %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %308 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x32x32xf32>) -> tensor<256xf32>
    %49 = "mhlo.fusion"(%48) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %50 = "mhlo.fusion"(%47, %49) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x2x2x16x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x128x2x2x16x16xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x128x2x2x16x16xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %294 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x2x2x16x16xf32>, tensor<f32>) -> tensor<256xf32>
    %51 = "mhlo.fusion"(%50) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %52 = "mhlo.fusion"(%arg43, %arg44, %47, %51, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x2x2x16x16xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %289 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %290 = stablehlo.multiply %289, %cst : tensor<1xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
      %293 = stablehlo.subtract %arg247, %292 : tensor<1x128x2x2x16x16xf32>
      %294 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [] : (tensor<f32>) -> tensor<1x128x2x2x16x16xf32>
      %300 = stablehlo.multiply %293, %299 : tensor<1x128x2x2x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x1x131072xf32>
      %302 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<1x1x131072xf32>
      %305 = stablehlo.add %288, %304 : tensor<1x1x131072xf32>
      mhlo.return %305 : tensor<1x1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x2x2x16x16xf32>, tensor<f32>, tensor<f32>) -> tensor<1x1x131072xf32>
    %53 = mhlo.bitcast %52 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
    %54 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%53, %arg45, %arg46) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<1x257x4x256xf32>, tensor<0xui8>>
    %55 = stablehlo.get_tuple_element %54[0] : (tuple<tensor<1x257x4x256xf32>, tensor<0xui8>>) -> tensor<1x257x4x256xf32>
    %56 = "mhlo.fusion"(%55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x257x4x256xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x128x4x256xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x256xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<128x4x256xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4x256xf32>, tensor<f32>) -> tensor<128x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x128x4x256xf32>
      mhlo.return %307 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %57 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x128x4x256xf32>
    %58 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%56, %arg47, %arg48, %57) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>) -> tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>
    %59 = stablehlo.get_tuple_element %58[0] : (tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>) -> tensor<1x128x4x256xf32>
    %60 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %288 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>) -> tensor<256xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %62 = "mhlo.fusion"(%59, %61) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x128x4x256xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %294 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<256xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %64 = "mhlo.fusion"(%arg49, %arg50, %59, %63, %61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x4x256xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x128x4x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x128x4x256xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x128x4x256xf32>
      mhlo.return %302 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>, tensor<f32>, tensor<f32>) -> tensor<1x128x4x256xf32>
    %65 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%64, %arg51, %arg52) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<1x256x4x256xf32>, tensor<0xui8>>
    %66 = stablehlo.get_tuple_element %65[0] : (tuple<tensor<1x256x4x256xf32>, tensor<0xui8>>) -> tensor<1x256x4x256xf32>
    %67 = "mhlo.fusion"(%66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x256x4x256xf32>
      %289 = stablehlo.exponential %288 : tensor<1x256x4x256xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x256x4x256xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x256x4x256xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x256x4x256xf32>
      mhlo.return %292 : tensor<1x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x256xf32>) -> tensor<1x256x4x256xf32>
    %68 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%67, %arg53, %arg54, %59) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>) -> tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>
    %69 = stablehlo.get_tuple_element %68[0] : (tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>) -> tensor<1x128x4x256xf32>
    %70 = "mhlo.fusion"(%69) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %288 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>) -> tensor<256xf32>
    %71 = "mhlo.fusion"(%70) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %72 = "mhlo.fusion"(%69, %71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x128x4x256xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %294 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<256xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %74 = "mhlo.fusion"(%arg55, %arg56, %69, %73, %71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x4x256xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x128x4x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x128x4x256xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x128x4x256xf32>
      mhlo.return %302 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>, tensor<f32>, tensor<f32>) -> tensor<1x128x4x256xf32>
    %75 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%74, %arg57, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<1x257x4x256xf32>, tensor<0xui8>>
    %76 = stablehlo.get_tuple_element %75[0] : (tuple<tensor<1x257x4x256xf32>, tensor<0xui8>>) -> tensor<1x257x4x256xf32>
    %77 = "mhlo.fusion"(%76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x257x4x256xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x128x4x256xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x256xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<128x4x256xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4x256xf32>, tensor<f32>) -> tensor<128x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x128x4x256xf32>
      mhlo.return %307 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
    %78 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%77, %arg59, %arg60, %69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>) -> tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>
    %79 = stablehlo.get_tuple_element %78[0] : (tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>) -> tensor<1x128x4x256xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %288 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>) -> tensor<256xf32>
    %81 = "mhlo.fusion"(%80) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %82 = "mhlo.fusion"(%79, %81) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x128x4x256xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %294 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<256xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %84 = "mhlo.fusion"(%arg61, %arg62, %79, %83, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x4x256xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x128x4x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x128x4x256xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x128x4x256xf32>
      mhlo.return %302 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>, tensor<f32>, tensor<f32>) -> tensor<1x128x4x256xf32>
    %85 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%84, %arg63, %arg64) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<1x256x4x256xf32>, tensor<0xui8>>
    %86 = stablehlo.get_tuple_element %85[0] : (tuple<tensor<1x256x4x256xf32>, tensor<0xui8>>) -> tensor<1x256x4x256xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x256x4x256xf32>
      %289 = stablehlo.exponential %288 : tensor<1x256x4x256xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x256x4x256xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x256x4x256xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x256x4x256xf32>
      mhlo.return %292 : tensor<1x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x256xf32>) -> tensor<1x256x4x256xf32>
    %88 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%87, %arg65, %arg66, %79) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>) -> tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>
    %89 = stablehlo.get_tuple_element %88[0] : (tuple<tensor<1x128x4x256xf32>, tensor<0xui8>>) -> tensor<1x128x4x256xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %288 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>) -> tensor<256xf32>
    %91 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %92 = "mhlo.fusion"(%89, %91) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x128x4x256xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x512xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %294 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<256xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<f32>
    %94 = "mhlo.fusion"(%arg67, %arg68, %89, %93, %91) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1x128x4x256xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x128x4x256xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x128x4x256xf32>
      %300 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x128x4x256xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x128x4x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x128x2x2x16x16xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x2x16x2x16x128xf32>
      mhlo.return %304 : tensor<1x2x16x2x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<1x128x4x256xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2x16x2x16x128xf32>
    %95 = "mhlo.fusion"(%2, %94) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x128x32x32xf32>, %arg246: tensor<1x2x16x2x16x128xf32>):
      %287 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %288 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %289 = stablehlo.add %287, %288 : tensor<2x16xi32>
      %290 = mhlo.bitcast %289 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %292 = stablehlo.compare LT, %290, %291 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %293 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %294 = stablehlo.add %290, %293 : tensor<1x1x2x16xi32>
      %295 = stablehlo.select %292, %294, %290 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %298 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %299 = stablehlo.compare LT, %289, %298 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %300 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %301 = stablehlo.add %289, %300 : tensor<2x16xi32>
      %302 = stablehlo.select %299, %301, %289 : tensor<2x16xi1>, tensor<2x16xi32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %304 = stablehlo.concatenate %297, %303, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %306 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x16x2x16x128xf32>) -> tensor<1024x1x128x1x1xf32>
      %307 = "stablehlo.scatter"(%arg245, %305, %306) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %308 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %308 : tensor<f32>
      }) : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x128x32x32xf32>
      mhlo.return %307 : tensor<1x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x32x32xf32>, tensor<1x2x16x2x16x128xf32>) -> tensor<1x128x32x32xf32>
    %96 = stablehlo.custom_call @__cudnn$convForward(%95, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x32x32xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>
    %97 = stablehlo.get_tuple_element %96[0] : (tuple<tensor<1x256x32x32xf32>, tensor<0xui8>>) -> tensor<1x256x32x32xf32>
    %98 = "mhlo.fusion"(%arg71, %arg70, %97, %arg222, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x256x32x32xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x256x32x32xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32x32xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x256x32x32xf32>
      mhlo.return %297 : tensor<1x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x32x32xf32>
    %99 = stablehlo.custom_call @__cudnn$convForward(%98, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x32x32xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<1x512x32x32xf32>, tensor<0xui8>>
    %100 = stablehlo.get_tuple_element %99[0] : (tuple<tensor<1x512x32x32xf32>, tensor<0xui8>>) -> tensor<1x512x32x32xf32>
    %101 = "mhlo.fusion"(%arg74, %arg73, %100, %arg224, %arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<1x512x32x32xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x512x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x512x32x32xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x512x32x32xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1x512x32x32xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x512x32x32xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x32x32xf32>
      %299 = stablehlo.negate %297 : tensor<1x512x32x32xf32>
      %300 = stablehlo.exponential %299 : tensor<1x512x32x32xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x512x32x32xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x512x32x32xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x512x32x32xf32>
      mhlo.return %303 : tensor<1x512x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x32x32xf32>
    %102 = stablehlo.custom_call @__cudnn$convForward(%101, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x32x32xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<1x512x16x16xf32>, tensor<0xui8>>
    %103 = stablehlo.get_tuple_element %102[0] : (tuple<tensor<1x512x16x16xf32>, tensor<0xui8>>) -> tensor<1x512x16x16xf32>
    %104 = "mhlo.fusion"(%arg77, %arg76, %103, %arg226, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<1x512x16x16xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x512x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x512x16x16xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x512x16x16xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1x512x16x16xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x512x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x16x16xf32>
      %299 = stablehlo.negate %297 : tensor<1x512x16x16xf32>
      %300 = stablehlo.exponential %299 : tensor<1x512x16x16xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x512x16x16xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x512x16x16xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x512x16x16xf32>
      mhlo.return %303 : tensor<1x512x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x16x16xf32>
    %105 = stablehlo.custom_call @__cudnn$convForward(%104, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x16x16xf32>, tensor<384x512x1x1xf32>) -> tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>
    %106 = stablehlo.get_tuple_element %105[0] : (tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>) -> tensor<1x384x16x16xf32>
    %107 = "mhlo.fusion"(%arg80, %arg79, %106, %arg228, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<1x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x384x16x16xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x384x16x16xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x384x16x16xf32>
      mhlo.return %297 : tensor<1x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %108 = stablehlo.custom_call @__cudnn$convForward(%107, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x16x16xf32>, tensor<384x1x3x3xf32>) -> tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>
    %109 = stablehlo.get_tuple_element %108[0] : (tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>) -> tensor<1x384x16x16xf32>
    %110 = "mhlo.fusion"(%arg83, %arg82, %109, %arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<1x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x384x16x16xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x384x16x16xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x384x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x384x16x16xf32>
      %299 = stablehlo.negate %297 : tensor<1x384x16x16xf32>
      %300 = stablehlo.exponential %299 : tensor<1x384x16x16xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x384x16x16xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x384x16x16xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x384x16x16xf32>
      mhlo.return %303 : tensor<1x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %111 = stablehlo.custom_call @__cudnn$convForward(%110, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x16x16xf32>, tensor<192x384x1x1xf32>) -> tuple<tensor<1x192x16x16xf32>, tensor<0xui8>>
    %112 = stablehlo.get_tuple_element %111[0] : (tuple<tensor<1x192x16x16xf32>, tensor<0xui8>>) -> tensor<1x192x16x16xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x16x16xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<8xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x8xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x8xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x8xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x8xi1>, tensor<2x8xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %309 = "stablehlo.gather"(%arg245, %308) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 192, 1, 1>}> : (tensor<1x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1x192x1x1xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<1x2x8x2x8x192xf32>
      %311 = stablehlo.transpose %310, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x8x2x8x192xf32>) -> tensor<1x192x2x2x8x8xf32>
      mhlo.return %311 : tensor<1x192x2x2x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x16x16xf32>) -> tensor<1x192x2x2x8x8xf32>
    %114 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x16x16xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<8xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x8xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x8xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x8xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x8xi1>, tensor<2x8xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %309 = "stablehlo.gather"(%arg245, %308) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 192, 1, 1>}> : (tensor<1x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1x192x1x1xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %311 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x16x16xf32>) -> tensor<128xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %116 = "mhlo.fusion"(%113, %115) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x2x2x8x8xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x2x2x8x8xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x2x2x8x8xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x2x2x8x8xf32>, tensor<f32>) -> tensor<128xf32>
    %117 = "mhlo.fusion"(%116) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %118 = "mhlo.fusion"(%arg86, %arg85, %113, %117, %115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x2x2x8x8xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %289 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %290 = stablehlo.multiply %289, %cst : tensor<1xf32>
      %291 = mhlo.bitcast %290 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
      %293 = stablehlo.subtract %arg247, %292 : tensor<1x192x2x2x8x8xf32>
      %294 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [] : (tensor<f32>) -> tensor<1x192x2x2x8x8xf32>
      %300 = stablehlo.multiply %293, %299 : tensor<1x192x2x2x8x8xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x1x49152xf32>
      %302 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<1x1x49152xf32>
      %305 = stablehlo.add %288, %304 : tensor<1x1x49152xf32>
      mhlo.return %305 : tensor<1x1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x2x2x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x1x49152xf32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
    %120 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%119, %arg87, %arg88) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>
    %121 = stablehlo.get_tuple_element %120[0] : (tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>) -> tensor<1x385x4x64xf32>
    %122 = "mhlo.fusion"(%121) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x385x4x64xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x192x4x64xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x64xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x64xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x192x4x64xf32>
      mhlo.return %307 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %123 = mhlo.bitcast %113 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x192x4x64xf32>
    %124 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%122, %arg89, %arg90, %123) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %125 = stablehlo.get_tuple_element %124[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %126 = "mhlo.fusion"(%125) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %128 = "mhlo.fusion"(%125, %127) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %130 = "mhlo.fusion"(%arg92, %arg91, %125, %129, %127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %131 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%130, %arg93, %arg94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>
    %132 = stablehlo.get_tuple_element %131[0] : (tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>) -> tensor<1x384x4x64xf32>
    %133 = "mhlo.fusion"(%132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x384x4x64xf32>
      %289 = stablehlo.exponential %288 : tensor<1x384x4x64xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x384x4x64xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x384x4x64xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x384x4x64xf32>
      mhlo.return %292 : tensor<1x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32>
    %134 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%133, %arg95, %arg96, %125) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %136 = "mhlo.fusion"(%135) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %137 = "mhlo.fusion"(%136) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %138 = "mhlo.fusion"(%135, %137) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %140 = "mhlo.fusion"(%arg98, %arg97, %135, %139, %137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %141 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%140, %arg99, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>
    %142 = stablehlo.get_tuple_element %141[0] : (tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>) -> tensor<1x385x4x64xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x385x4x64xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x192x4x64xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x64xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x64xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x192x4x64xf32>
      mhlo.return %307 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %144 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%143, %arg101, %arg102, %135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %145 = stablehlo.get_tuple_element %144[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %146 = "mhlo.fusion"(%145) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %147 = "mhlo.fusion"(%146) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %148 = "mhlo.fusion"(%145, %147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %150 = "mhlo.fusion"(%arg104, %arg103, %145, %149, %147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %151 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%150, %arg105, %arg106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>
    %152 = stablehlo.get_tuple_element %151[0] : (tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>) -> tensor<1x384x4x64xf32>
    %153 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x384x4x64xf32>
      %289 = stablehlo.exponential %288 : tensor<1x384x4x64xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x384x4x64xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x384x4x64xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x384x4x64xf32>
      mhlo.return %292 : tensor<1x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32>
    %154 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%153, %arg107, %arg108, %145) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %155 = stablehlo.get_tuple_element %154[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %156 = "mhlo.fusion"(%155) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %157 = "mhlo.fusion"(%156) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %158 = "mhlo.fusion"(%155, %157) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %160 = "mhlo.fusion"(%arg110, %arg109, %155, %159, %157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %161 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%160, %arg111, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>
    %162 = stablehlo.get_tuple_element %161[0] : (tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>) -> tensor<1x385x4x64xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x385x4x64xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x192x4x64xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x64xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x64xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x192x4x64xf32>
      mhlo.return %307 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %164 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%163, %arg113, %arg114, %155) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %165 = stablehlo.get_tuple_element %164[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %166 = "mhlo.fusion"(%165) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %167 = "mhlo.fusion"(%166) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %168 = "mhlo.fusion"(%165, %167) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %170 = "mhlo.fusion"(%arg116, %arg115, %165, %169, %167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %171 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%170, %arg117, %arg118) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>
    %172 = stablehlo.get_tuple_element %171[0] : (tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>) -> tensor<1x384x4x64xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x384x4x64xf32>
      %289 = stablehlo.exponential %288 : tensor<1x384x4x64xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x384x4x64xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x384x4x64xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x384x4x64xf32>
      mhlo.return %292 : tensor<1x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32>
    %174 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%173, %arg119, %arg120, %165) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %175 = stablehlo.get_tuple_element %174[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %176 = "mhlo.fusion"(%175) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %178 = "mhlo.fusion"(%175, %177) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %180 = "mhlo.fusion"(%arg122, %arg121, %175, %179, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %181 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%180, %arg123, %arg124) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>
    %182 = stablehlo.get_tuple_element %181[0] : (tuple<tensor<1x385x4x64xf32>, tensor<0xui8>>) -> tensor<1x385x4x64xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x385x4x64xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x192x4x64xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x64xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x64xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x64xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<192x4x64xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<192x4x64xf32>, tensor<f32>) -> tensor<192x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x192x4x64xf32>
      mhlo.return %307 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
    %184 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%183, %arg125, %arg126, %175) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %185 = stablehlo.get_tuple_element %184[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %186 = "mhlo.fusion"(%185) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %188 = "mhlo.fusion"(%185, %187) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %190 = "mhlo.fusion"(%arg128, %arg127, %185, %189, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      mhlo.return %302 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x192x4x64xf32>
    %191 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%190, %arg129, %arg130) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>
    %192 = stablehlo.get_tuple_element %191[0] : (tuple<tensor<1x384x4x64xf32>, tensor<0xui8>>) -> tensor<1x384x4x64xf32>
    %193 = "mhlo.fusion"(%192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x384x4x64xf32>
      %289 = stablehlo.exponential %288 : tensor<1x384x4x64xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x384x4x64xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x384x4x64xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x384x4x64xf32>
      mhlo.return %292 : tensor<1x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x4x64xf32>) -> tensor<1x384x4x64xf32>
    %194 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%193, %arg131, %arg132, %185) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>) -> tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>
    %195 = stablehlo.get_tuple_element %194[0] : (tuple<tensor<1x192x4x64xf32>, tensor<0xui8>>) -> tensor<1x192x4x64xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %288 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>) -> tensor<128xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %198 = "mhlo.fusion"(%195, %197) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x192x4x64xf32>
      %292 = stablehlo.multiply %291, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x384xf32>, tensor<f32>) -> tensor<128xf32>
      mhlo.return %294 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<128xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %287 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %287 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<f32>
    %200 = "mhlo.fusion"(%arg133, %arg134, %195, %199, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<1x192x4x64xf32>, %arg248: tensor<f32>, %arg249: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %288 = mhlo.bitcast %arg249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %292 = stablehlo.subtract %arg247, %291 : tensor<1x192x4x64xf32>
      %293 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %294 = stablehlo.multiply %293, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %295 = stablehlo.add %294, %cst_0 : tensor<1xf32>
      %296 = stablehlo.rsqrt %295 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %299 = stablehlo.multiply %292, %298 : tensor<1x192x4x64xf32>
      %300 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %301 = stablehlo.multiply %299, %300 : tensor<1x192x4x64xf32>
      %302 = stablehlo.add %287, %301 : tensor<1x192x4x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x192x2x2x8x8xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x2x8x2x8x192xf32>
      mhlo.return %304 : tensor<1x2x8x2x8x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x4x64xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2x8x2x8x192xf32>
    %201 = "mhlo.fusion"(%1, %200) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x192x16x16xf32>, %arg246: tensor<1x2x8x2x8x192xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<8xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x8xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x8xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x8xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x8xi1>, tensor<2x8xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %309 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x8x2x8x192xf32>) -> tensor<256x1x192x1x1xf32>
      %310 = "stablehlo.scatter"(%arg245, %308, %309) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %311 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %311 : tensor<f32>
      }) : (tensor<1x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1x192x1x1xf32>) -> tensor<1x192x16x16xf32>
      mhlo.return %310 : tensor<1x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x16x16xf32>, tensor<1x2x8x2x8x192xf32>) -> tensor<1x192x16x16xf32>
    %202 = stablehlo.custom_call @__cudnn$convForward(%201, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x16x16xf32>, tensor<384x192x1x1xf32>) -> tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>
    %203 = stablehlo.get_tuple_element %202[0] : (tuple<tensor<1x384x16x16xf32>, tensor<0xui8>>) -> tensor<1x384x16x16xf32>
    %204 = "mhlo.fusion"(%arg137, %arg136, %203, %arg232, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<1x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x384x16x16xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x384x16x16xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1x384x16x16xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x384x16x16xf32>
      mhlo.return %297 : tensor<1x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x384x16x16xf32>
    %205 = stablehlo.custom_call @__cudnn$convForward(%204, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x16x16xf32>, tensor<768x384x1x1xf32>) -> tuple<tensor<1x768x16x16xf32>, tensor<0xui8>>
    %206 = stablehlo.get_tuple_element %205[0] : (tuple<tensor<1x768x16x16xf32>, tensor<0xui8>>) -> tensor<1x768x16x16xf32>
    %207 = "mhlo.fusion"(%arg140, %arg139, %206, %arg234, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<1x768x16x16xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x768x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x768x16x16xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x768x16x16xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<1x768x16x16xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x768x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768x16x16xf32>
      %299 = stablehlo.negate %297 : tensor<1x768x16x16xf32>
      %300 = stablehlo.exponential %299 : tensor<1x768x16x16xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x768x16x16xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x768x16x16xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x768x16x16xf32>
      mhlo.return %303 : tensor<1x768x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x768x16x16xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x768x16x16xf32>
    %208 = stablehlo.custom_call @__cudnn$convForward(%207, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1x768x16x16xf32>, tensor<768x1x3x3xf32>) -> tuple<tensor<1x768x8x8xf32>, tensor<0xui8>>
    %209 = stablehlo.get_tuple_element %208[0] : (tuple<tensor<1x768x8x8xf32>, tensor<0xui8>>) -> tensor<1x768x8x8xf32>
    %210 = "mhlo.fusion"(%arg143, %arg142, %209, %arg236, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<1x768x8x8xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x768x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x768x8x8xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x768x8x8xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<1x768x8x8xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x768x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768x8x8xf32>
      %299 = stablehlo.negate %297 : tensor<1x768x8x8xf32>
      %300 = stablehlo.exponential %299 : tensor<1x768x8x8xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x768x8x8xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x768x8x8xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x768x8x8xf32>
      mhlo.return %303 : tensor<1x768x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x768x8x8xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x768x8x8xf32>
    %211 = stablehlo.custom_call @__cudnn$convForward(%210, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1x768x8x8xf32>, tensor<512x768x1x1xf32>) -> tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>
    %212 = stablehlo.get_tuple_element %211[0] : (tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>) -> tensor<1x512x8x8xf32>
    %213 = "mhlo.fusion"(%arg146, %arg145, %212, %arg238, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<1x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x512x8x8xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x512x8x8xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x512x8x8xf32>
      mhlo.return %297 : tensor<1x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %214 = stablehlo.custom_call @__cudnn$convForward(%213, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x8x8xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>
    %215 = stablehlo.get_tuple_element %214[0] : (tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>) -> tensor<1x512x8x8xf32>
    %216 = "mhlo.fusion"(%arg149, %arg148, %215, %arg240, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<1x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x512x8x8xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x512x8x8xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x512x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x512x8x8xf32>
      %299 = stablehlo.negate %297 : tensor<1x512x8x8xf32>
      %300 = stablehlo.exponential %299 : tensor<1x512x8x8xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x512x8x8xf32>
      %302 = stablehlo.divide %298, %301 : tensor<1x512x8x8xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x512x8x8xf32>
      mhlo.return %303 : tensor<1x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x8x8xf32>
    %217 = stablehlo.custom_call @__cudnn$convForward(%216, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x8x8xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<1x256x8x8xf32>, tensor<0xui8>>
    %218 = stablehlo.get_tuple_element %217[0] : (tuple<tensor<1x256x8x8xf32>, tensor<0xui8>>) -> tensor<1x256x8x8xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x8x8xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<4xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x4xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x4xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x4xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x4xi1>, tensor<2x4xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %309 = "stablehlo.gather"(%arg245, %308) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1x256x1x1xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<1x2x4x2x4x256xf32>
      %311 = stablehlo.transpose %310, dims = [0, 5, 1, 3, 2, 4] : (tensor<1x2x4x2x4x256xf32>) -> tensor<1x256x2x2x4x4xf32>
      mhlo.return %311 : tensor<1x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8x8xf32>) -> tensor<1x256x2x2x4x4xf32>
    %220 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x8x8xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<4xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x4xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x4xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x4xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x4xi1>, tensor<2x4xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %309 = "stablehlo.gather"(%arg245, %308) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1x256x1x1xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %311 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8x8xf32>) -> tensor<f32>
    %221 = "mhlo.fusion"(%219, %220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x2x2x4x4xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x2x2x4x4xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x2x2x4x4xf32>
      mhlo.return %291 : tensor<1x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2x2x4x4xf32>, tensor<f32>) -> tensor<1x256x2x2x4x4xf32>
    %222 = "mhlo.fusion"(%221) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x2x2x4x4xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x2x2x4x4xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2x2x4x4xf32>) -> tensor<f32>
    %223 = "mhlo.fusion"(%221, %arg151, %arg152, %222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x2x2x4x4xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %289 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %290 = stablehlo.multiply %289, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %291 = stablehlo.add %290, %cst_0 : tensor<1xf32>
      %292 = stablehlo.rsqrt %291 : tensor<1xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [] : (tensor<f32>) -> tensor<1x256x2x2x4x4xf32>
      %295 = stablehlo.multiply %arg245, %294 : tensor<1x256x2x2x4x4xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x1x16384xf32>
      %297 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %299 = stablehlo.multiply %296, %298 : tensor<1x1x16384xf32>
      %300 = stablehlo.add %288, %299 : tensor<1x1x16384xf32>
      mhlo.return %300 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2x2x4x4xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %224 = mhlo.bitcast %223 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
    %225 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%224, %arg153, %arg154) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>
    %226 = stablehlo.get_tuple_element %225[0] : (tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>) -> tensor<1x513x4x16xf32>
    %227 = "mhlo.fusion"(%226) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x513x4x16xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x256x4x16xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x16xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x16xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x256x4x16xf32>
      mhlo.return %307 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %228 = mhlo.bitcast %219 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x256x4x16xf32>
    %229 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%227, %arg155, %arg156, %228) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %230 = stablehlo.get_tuple_element %229[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %232 = "mhlo.fusion"(%230, %231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %234 = "mhlo.fusion"(%232, %arg157, %arg158, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg245, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      mhlo.return %297 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %235 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%234, %arg159, %arg160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>
    %236 = stablehlo.get_tuple_element %235[0] : (tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>) -> tensor<1x512x4x16xf32>
    %237 = "mhlo.fusion"(%236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x512x4x16xf32>
      %289 = stablehlo.exponential %288 : tensor<1x512x4x16xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x512x4x16xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x512x4x16xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x512x4x16xf32>
      mhlo.return %292 : tensor<1x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32>
    %238 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%237, %arg161, %arg162, %230) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %239 = stablehlo.get_tuple_element %238[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %241 = "mhlo.fusion"(%239, %240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %242 = "mhlo.fusion"(%241) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %243 = "mhlo.fusion"(%241, %arg163, %arg164, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg245, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      mhlo.return %297 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %244 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%243, %arg165, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>
    %245 = stablehlo.get_tuple_element %244[0] : (tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>) -> tensor<1x513x4x16xf32>
    %246 = "mhlo.fusion"(%245) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x513x4x16xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x256x4x16xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x16xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x16xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x256x4x16xf32>
      mhlo.return %307 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %247 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%246, %arg167, %arg168, %239) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %248 = stablehlo.get_tuple_element %247[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %249 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %250 = "mhlo.fusion"(%248, %249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %251 = "mhlo.fusion"(%250) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %252 = "mhlo.fusion"(%250, %arg169, %arg170, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg245, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      mhlo.return %297 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %253 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%252, %arg171, %arg172) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>
    %254 = stablehlo.get_tuple_element %253[0] : (tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>) -> tensor<1x512x4x16xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x512x4x16xf32>
      %289 = stablehlo.exponential %288 : tensor<1x512x4x16xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x512x4x16xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x512x4x16xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x512x4x16xf32>
      mhlo.return %292 : tensor<1x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32>
    %256 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%255, %arg173, %arg174, %248) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %257 = stablehlo.get_tuple_element %256[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %259 = "mhlo.fusion"(%257, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %260 = "mhlo.fusion"(%259) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %261 = "mhlo.fusion"(%259, %arg175, %arg176, %260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg245, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      mhlo.return %297 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %262 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%261, %arg177, %arg178) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>
    %263 = stablehlo.get_tuple_element %262[0] : (tuple<tensor<1x513x4x16xf32>, tensor<0xui8>>) -> tensor<1x513x4x16xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x513x4x16xf32>):
      %287 = stablehlo.slice %arg245 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %289 = stablehlo.maximum %287, %288 : tensor<1x256x4x16xf32>
      %290 = stablehlo.slice %arg245 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.slice %arg245 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %294 = stablehlo.broadcast_in_dim %293, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %295 = stablehlo.subtract %291, %294 : tensor<1x1x4x16xf32>
      %296 = stablehlo.exponential %295 : tensor<1x1x4x16xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %300 = stablehlo.divide %296, %299 : tensor<1x1x4x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %303 = stablehlo.multiply %290, %302 : tensor<1x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<256x4x16xf32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x4x16xf32>, tensor<f32>) -> tensor<256x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %307 = stablehlo.multiply %289, %306 : tensor<1x256x4x16xf32>
      mhlo.return %307 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
    %265 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%264, %arg179, %arg180, %257) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %266 = stablehlo.get_tuple_element %265[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %267 = "mhlo.fusion"(%266) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %268 = "mhlo.fusion"(%266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %269 = "mhlo.fusion"(%268) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %270 = "mhlo.fusion"(%268, %arg181, %arg182, %269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg245, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      mhlo.return %297 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %271 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%270, %arg183, %arg184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>
    %272 = stablehlo.get_tuple_element %271[0] : (tuple<tensor<1x512x4x16xf32>, tensor<0xui8>>) -> tensor<1x512x4x16xf32>
    %273 = "mhlo.fusion"(%272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %287 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %288 = stablehlo.negate %arg245 : tensor<1x512x4x16xf32>
      %289 = stablehlo.exponential %288 : tensor<1x512x4x16xf32>
      %290 = stablehlo.add %287, %289 : tensor<1x512x4x16xf32>
      %291 = stablehlo.divide %287, %290 : tensor<1x512x4x16xf32>
      %292 = stablehlo.multiply %arg245, %291 : tensor<1x512x4x16xf32>
      mhlo.return %292 : tensor<1x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x4x16xf32>) -> tensor<1x512x4x16xf32>
    %274 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%273, %arg185, %arg186, %266) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1x256x4x16xf32>) -> tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>
    %275 = stablehlo.get_tuple_element %274[0] : (tuple<tensor<1x256x4x16xf32>, tensor<0xui8>>) -> tensor<1x256x4x16xf32>
    %276 = "mhlo.fusion"(%275) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = mhlo.bitcast %arg245 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.reduce(%287 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %288 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %277 = "mhlo.fusion"(%275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<f32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %288 = stablehlo.multiply %287, %cst : tensor<1xf32>
      %289 = mhlo.bitcast %288 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %291 = stablehlo.subtract %arg245, %290 : tensor<1x256x4x16xf32>
      mhlo.return %291 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4x16xf32>
    %278 = "mhlo.fusion"(%277) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>):
      %287 = stablehlo.multiply %arg245, %arg245 : tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %287 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.reduce(%288 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %289 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>) -> tensor<f32>
    %279 = "mhlo.fusion"(%arg187, %277, %arg188, %278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1x256x4x16xf32>, %arg247: tensor<256xf32>, %arg248: tensor<f32>):
      %287 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %288 = mhlo.bitcast %arg248 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %289 = stablehlo.multiply %288, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %290 = stablehlo.add %289, %cst_0 : tensor<1xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %294 = stablehlo.multiply %arg246, %293 : tensor<1x256x4x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x256x4x16xf32>
      %297 = stablehlo.add %287, %296 : tensor<1x256x4x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x256x2x2x4x4xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 4, 3, 5, 1] : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x2x4x2x4x256xf32>
      mhlo.return %299 : tensor<1x2x4x2x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>, tensor<f32>) -> tensor<1x2x4x2x4x256xf32>
    %280 = "mhlo.fusion"(%0, %279) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1x256x8x8xf32>, %arg246: tensor<1x2x4x2x4x256xf32>):
      %287 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %288 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %289 = stablehlo.multiply %287, %288 : tensor<4xi32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %291 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %292 = stablehlo.add %290, %291 : tensor<2x4xi32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %295 = stablehlo.compare LT, %293, %294 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %296 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %297 = stablehlo.add %293, %296 : tensor<1x1x2x4xi32>
      %298 = stablehlo.select %295, %297, %293 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %301 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %302 = stablehlo.compare LT, %292, %301 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %303 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %304 = stablehlo.add %292, %303 : tensor<2x4xi32>
      %305 = stablehlo.select %302, %304, %292 : tensor<2x4xi1>, tensor<2x4xi32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %307 = stablehlo.concatenate %300, %306, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %309 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x2x4x2x4x256xf32>) -> tensor<64x1x256x1x1xf32>
      %310 = "stablehlo.scatter"(%arg245, %308, %309) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %311 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %311 : tensor<f32>
      }) : (tensor<1x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1x256x1x1xf32>) -> tensor<1x256x8x8xf32>
      mhlo.return %310 : tensor<1x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8x8xf32>, tensor<1x2x4x2x4x256xf32>) -> tensor<1x256x8x8xf32>
    %281 = stablehlo.custom_call @__cudnn$convForward(%280, %arg189) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x8x8xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>
    %282 = stablehlo.get_tuple_element %281[0] : (tuple<tensor<1x512x8x8xf32>, tensor<0xui8>>) -> tensor<1x512x8x8xf32>
    %283 = "mhlo.fusion"(%arg191, %arg190, %282, %arg242, %arg243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<1x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %287 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %288 = stablehlo.subtract %arg247, %287 : tensor<1x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg249, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      %292 = stablehlo.broadcast_in_dim %291, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %293 = stablehlo.multiply %288, %292 : tensor<1x512x8x8xf32>
      %294 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %295 = stablehlo.multiply %293, %294 : tensor<1x512x8x8xf32>
      %296 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1x512x8x8xf32>
      %297 = stablehlo.add %295, %296 : tensor<1x512x8x8xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x512x8x8xf32>) -> tensor<512x64xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%298 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<512x64xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %299 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<1x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %284 = "mhlo.fusion"(%arg192, %283) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<2x512xf32>, %arg246: tensor<512xf32>):
      %287 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x512xf32>
      %289 = stablehlo.multiply %287, %288 : tensor<1x512xf32>
      %290 = mhlo.bitcast %289 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x512xf32>) -> tensor<512xf32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [1] : (tensor<512xf32>) -> tensor<2x512xf32>
      %292 = stablehlo.multiply %291, %arg245 : tensor<2x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2x512xf32>, tensor<f32>) -> tensor<2xf32>
      mhlo.return %293 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2xf32>
    %285 = "mhlo.fusion"(%arg193, %284) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<2xf32>, %arg246: tensor<2xf32>):
      %287 = stablehlo.add %arg245, %arg246 : tensor<2xf32>
      mhlo.return %287 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %286 = mhlo.bitcast %285 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2xf32>) -> tensor<1x2xf32>
    return %286 : tensor<1x2xf32>
  }
}
