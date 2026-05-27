module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_5.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<1x8x49x196xf32>, %arg1: tensor<f32>) -> tensor<1x8x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    return %0 : tensor<1x8x49xf32>
  }
  func.func private @region_3.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<f32>) -> tensor<1x4x196xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    return %0 : tensor<1x4x196xf32>
  }
  func.func private @region_1.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<f32>) -> tensor<1x4x196xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    return %0 : tensor<1x4x196xf32>
  }
  func.func private @fused_computation.3(%arg0: tensor<1000xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384x1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.dot %arg1, %arg2, precision = [DEFAULT, DEFAULT] : (tensor<384xf32>, tensor<384x1000xf32>) -> tensor<1000xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func.func private @fused_computation.4(%arg0: tensor<1000x384xf32>) -> tensor<384x1000xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,1000]{0,1}"} : (tensor<1000x384xf32>) -> tensor<384x1000xf32>
    %1 = mhlo.copy %0 : tensor<384x1000xf32>
    return %1 : tensor<384x1000xf32>
  }
  func.func private @region_22.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>, %arg6: tensor<384xf32>, %arg7: tensor<384xf32>, %arg8: tensor<16x384xf32>, %arg9: tensor<384xf32>) -> tensor<384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg4, %7 : tensor<16x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<16x384xf32>, tensor<f32>) -> tensor<384xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-02> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<384xf32>
    %12 = stablehlo.subtract %11, %arg3 : tensor<384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %14 = stablehlo.add %arg2, %13 : tensor<384xf32>
    %15 = stablehlo.rsqrt %14 : tensor<384xf32>
    %16 = stablehlo.multiply %12, %15 : tensor<384xf32>
    %17 = stablehlo.multiply %16, %arg1 : tensor<384xf32>
    %18 = stablehlo.add %17, %arg0 : tensor<384xf32>
    return %18 : tensor<384xf32>
  }
  func.func private @fused_computation.6(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.7(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x16x768xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x16x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    return %17 : tensor<16x768xf32>
  }
  func.func private @fused_computation.8(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.9(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.10(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.11(%arg0: tensor<8x16x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
    %2 = mhlo.copy %1 : tensor<1x16x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x16x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x16x256xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    return %12 : tensor<16x256xf32>
  }
  func.func private @fused_computation.12(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
    return %12 : tensor<8x16x32xf32>
  }
  func.func private @fused_computation.13(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<1x8x16xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x8x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %3 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.14(%arg0: tensor<1x8x16xf32>, %arg1: tensor<256x8x1xf32>, %arg2: tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x8x16xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x8x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<1x8x16x16xf32>
    return %12 : tensor<1x8x16x16xf32>
  }
  func.func private @region_20.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<256x8x1xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<1x8x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %7 : tensor<1x8x16xf32>
  }
  func.func private @fused_computation.16(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = mhlo.copy %5 : tensor<256x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %7 : tensor<256x8x1xf32>
  }
  func.func private @fused_computation.17(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.18(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.19(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.20(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.21(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.22(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x16x768xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x16x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    return %17 : tensor<16x768xf32>
  }
  func.func private @fused_computation.23(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.24(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.25(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.26(%arg0: tensor<8x16x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
    %2 = mhlo.copy %1 : tensor<1x16x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x16x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x16x256xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    return %12 : tensor<16x256xf32>
  }
  func.func private @fused_computation.27(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
    return %12 : tensor<8x16x32xf32>
  }
  func.func private @fused_computation.28(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<1x8x16xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x8x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %3 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.29(%arg0: tensor<1x8x16xf32>, %arg1: tensor<256x8x1xf32>, %arg2: tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x8x16xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x8x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<1x8x16x16xf32>
    return %12 : tensor<1x8x16x16xf32>
  }
  func.func private @region_18.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<256x8x1xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<1x8x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %7 : tensor<1x8x16xf32>
  }
  func.func private @fused_computation.31(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = mhlo.copy %5 : tensor<256x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %7 : tensor<256x8x1xf32>
  }
  func.func private @fused_computation.32(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.33(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.34(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.35(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.36(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x16x768xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x16x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    return %17 : tensor<16x768xf32>
  }
  func.func private @fused_computation.38(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.40(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<8x16x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
    %2 = mhlo.copy %1 : tensor<1x16x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x16x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x16x256xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    return %12 : tensor<16x256xf32>
  }
  func.func private @fused_computation.42(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
    return %12 : tensor<8x16x32xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<1x8x16xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x8x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %3 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<1x8x16xf32>, %arg1: tensor<256x8x1xf32>, %arg2: tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x8x16xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x8x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<1x8x16x16xf32>
    return %12 : tensor<1x8x16x16xf32>
  }
  func.func private @region_16.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<256x8x1xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<1x8x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %7 : tensor<1x8x16xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = mhlo.copy %5 : tensor<256x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %7 : tensor<256x8x1xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.51(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x16x768xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x16x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    return %17 : tensor<16x768xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<16x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<16x384xf32>, %arg5: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x384xf32>
    return %8 : tensor<16x384xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<8x16x32xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
    %2 = mhlo.copy %1 : tensor<1x16x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x16x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x16x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x16x256xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
    return %12 : tensor<16x256xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
    return %12 : tensor<8x16x32xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<1x8x16xf32>) -> tensor<8x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x8x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %3 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<1x8x16xf32>, %arg1: tensor<256x8x1xf32>, %arg2: tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x8x16xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x8x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<1x8x16x16xf32>
    return %12 : tensor<1x8x16x16xf32>
  }
  func.func private @region_14.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.60(%arg0: tensor<256x8x1xf32>, %arg1: tensor<8x16x16xf32>) -> tensor<1x8x16xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8x16x16xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
    %4 = mhlo.copy %3 : tensor<8x16x16xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %7 : tensor<1x8x16xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xi32>) -> tensor<256x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
    %c_0 = stablehlo.constant dense<16> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x16xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x16xi1>, tensor<16x16xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
    %6 = mhlo.copy %5 : tensor<256x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
    return %7 : tensor<256x8x1xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512xf32>, %arg4: tensor<512xf32>) -> tensor<8x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
    return %12 : tensor<8x16x16xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x384xf32>, %arg4: tensor<384xf32>, %arg5: tensor<384xf32>, %arg6: tensor<384xf32>, %arg7: tensor<384xf32>, %arg8: tensor<16x384xf32>, %arg9: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<16x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16x384xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %15 = stablehlo.add %13, %14 : tensor<16x384xf32>
    %16 = stablehlo.add %7, %15 : tensor<16x384xf32>
    return %16 : tensor<16x384xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x16x768xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x16x768xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x16x768xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x16x768xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
    return %17 : tensor<16x768xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x384xf32>, %arg4: tensor<384xf32>) -> tensor<16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x384xf32>
    return %7 : tensor<16x384xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<16x16x64xf32>) -> tensor<16x1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x64xf32>) -> tensor<1x16x16x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,64]{3,1,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x16x16x64xf32>
    %2 = mhlo.copy %1 : tensor<1x16x16x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x64xf32>) -> tensor<1x16x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x16x1024xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x16x1024xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x16x1024xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x16x1024xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x1024xf32>) -> tensor<16x1024xf32>
    return %12 : tensor<16x1024xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<1280xf32>, %arg1: tensor<1280xf32>, %arg2: tensor<1280xf32>, %arg3: tensor<49x1280xf32>, %arg4: tensor<1280xf32>) -> tensor<16x49x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x1280xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x1280xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x1280xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x1280xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:16, 16:80] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,49,64]{3,1,2,0}"} : (tensor<1x49x16x64xf32>) -> tensor<1x16x49x64xf32>
    %11 = mhlo.copy %10 : tensor<1x16x49x64xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x49x64xf32>) -> tensor<16x49x64xf32>
    return %12 : tensor<16x49x64xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<1x16x16x49xf32>, %arg1: tensor<1x16x16xf32>) -> tensor<16x16x49xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x16x16x49xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
    return %3 : tensor<16x16x49xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<1x16x16xf32>, %arg1: tensor<784x16x1xf32>, %arg2: tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<16x16x49xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
    %4 = mhlo.copy %3 : tensor<16x16x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<16x16x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x16x16xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x16x16x49xf32>
    %12 = stablehlo.exponential %11 : tensor<1x16x16x49xf32>
    return %12 : tensor<1x16x16x49xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<784x16x1xf32>, %arg1: tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x16x49xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
    %4 = mhlo.copy %3 : tensor<16x16x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<16x16x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    return %6 : tensor<1x16x16x49xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<16x49xf32>, %arg1: tensor<16x49xi32>) -> tensor<784x16x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<16x49xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<16x49xi1>, tensor<16x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[784,1]{0,1}"} : (tensor<16x49xi32>) -> tensor<784x1xi32>
    %6 = mhlo.copy %5 : tensor<784x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 1>}> : (tensor<16x49xf32>, tensor<784x1xi32>) -> tensor<784x16x1xf32>
    return %7 : tensor<784x16x1xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<1280xf32>, %arg1: tensor<1280xf32>, %arg2: tensor<1280xf32>, %arg3: tensor<49x1280xf32>, %arg4: tensor<1280xf32>) -> tensor<16x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x1280xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x1280xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x1280xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x1280xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:16, 0:16] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,49]{2,1,3,0}"} : (tensor<1x49x16x16xf32>) -> tensor<1x16x16x49xf32>
    %11 = mhlo.copy %10 : tensor<1x16x16x49xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
    return %12 : tensor<16x16x49xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<1280xf32>) -> tensor<1280xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1280xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1280xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1280xf32>
    return %2 : tensor<1280xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256xf32>, %arg4: tensor<256xf32>) -> tensor<16x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x256xf32>) -> tensor<1x16x16x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,16]{3,1,2,0}"} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %10 = mhlo.copy %9 : tensor<1x16x16x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x16xf32>) -> tensor<16x16x16xf32>
    return %11 : tensor<16x16x16xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<49x256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x256xf32>) -> tensor<1x7x7x256xf32>
    %1 = stablehlo.slice %0 [0:1, 0:7:2, 0:7, 0:256] : (tensor<1x7x7x256xf32>) -> tensor<1x4x7x256xf32>
    %2 = stablehlo.slice %1 [0:1, 0:4, 0:7:2, 0:256] : (tensor<1x4x7x256xf32>) -> tensor<1x4x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x4x256xf32>) -> tensor<16x256xf32>
    return %3 : tensor<16x256xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<49x512xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x49x512xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x49x512xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    return %17 : tensor<49x512xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<6x49x32xf32>) -> tensor<49x192xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
    %2 = mhlo.copy %1 : tensor<1x49x6x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x49x192xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x49x192xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x49x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    return %12 : tensor<49x192xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
    return %12 : tensor<6x49x32xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<1x6x49xf32>) -> tensor<6x49x49xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x6x49x49xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
    return %3 : tensor<6x49x49xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<1x6x49xf32>, %arg1: tensor<2401x6x1xf32>, %arg2: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x6x49xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x6x49x49xf32>
    %12 = stablehlo.exponential %11 : tensor<1x6x49x49xf32>
    return %12 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<2401x6x1xf32>, %arg1: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    return %6 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<6x49xf32>, %arg1: tensor<49x49xi32>) -> tensor<2401x6x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = mhlo.copy %5 : tensor<2401x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    return %7 : tensor<2401x6x1xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
    %11 = mhlo.copy %10 : tensor<1x6x16x49xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    return %12 : tensor<6x16x49xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
    return %12 : tensor<6x49x16xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<49x512xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x49x512xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x49x512xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    return %17 : tensor<49x512xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<6x49x32xf32>) -> tensor<49x192xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
    %2 = mhlo.copy %1 : tensor<1x49x6x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x49x192xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x49x192xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x49x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    return %12 : tensor<49x192xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
    return %12 : tensor<6x49x32xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<1x6x49xf32>) -> tensor<6x49x49xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x6x49x49xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
    return %3 : tensor<6x49x49xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<1x6x49xf32>, %arg1: tensor<2401x6x1xf32>, %arg2: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x6x49xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x6x49x49xf32>
    %12 = stablehlo.exponential %11 : tensor<1x6x49x49xf32>
    return %12 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<2401x6x1xf32>, %arg1: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    return %6 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<6x49xf32>, %arg1: tensor<49x49xi32>) -> tensor<2401x6x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = mhlo.copy %5 : tensor<2401x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    return %7 : tensor<2401x6x1xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
    %11 = mhlo.copy %10 : tensor<1x6x16x49xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    return %12 : tensor<6x16x49xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
    return %12 : tensor<6x49x16xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<49x512xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x49x512xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x49x512xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    return %17 : tensor<49x512xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<49x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<49x256xf32>, %arg5: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<49x256xf32>
    return %8 : tensor<49x256xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<6x49x32xf32>) -> tensor<49x192xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
    %2 = mhlo.copy %1 : tensor<1x49x6x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x49x192xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x49x192xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x49x192xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x49x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
    return %12 : tensor<49x192xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
    return %12 : tensor<6x49x32xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<1x6x49xf32>) -> tensor<6x49x49xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x6x49x49xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
    return %3 : tensor<6x49x49xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<1x6x49xf32>, %arg1: tensor<2401x6x1xf32>, %arg2: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x6x49xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x6x49x49xf32>
    %12 = stablehlo.exponential %11 : tensor<1x6x49x49xf32>
    return %12 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<2401x6x1xf32>, %arg1: tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<6x49x49xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
    %4 = mhlo.copy %3 : tensor<6x49x49xf32>
    %5 = stablehlo.add %1, %4 : tensor<6x49x49xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    return %6 : tensor<1x6x49x49xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<6x49xf32>, %arg1: tensor<49x49xi32>) -> tensor<2401x6x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
    %c_0 = stablehlo.constant dense<49> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<49x49xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<49x49xi1>, tensor<49x49xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
    %6 = mhlo.copy %5 : tensor<2401x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
    return %7 : tensor<2401x6x1xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x16x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
    %11 = mhlo.copy %10 : tensor<1x6x16x49xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
    return %12 : tensor<6x16x49xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<49x384xf32>, %arg4: tensor<384xf32>) -> tensor<6x49x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
    %11 = mhlo.copy %10 : tensor<1x6x49x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
    return %12 : tensor<6x49x16xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<49x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>, %arg8: tensor<49x256xf32>, %arg9: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<49x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<49x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<49x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<49x256xf32>
    %16 = stablehlo.add %7, %15 : tensor<49x256xf32>
    return %16 : tensor<49x256xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<49x512xf32>, %arg4: tensor<512xf32>) -> tensor<49x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x512xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x49x512xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x49x512xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    return %17 : tensor<49x512xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<49x256xf32>, %arg4: tensor<256xf32>) -> tensor<49x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x256xf32>
    return %7 : tensor<49x256xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<8x49x64xf32>) -> tensor<49x512xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x64xf32>) -> tensor<1x8x49x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,8,64]{3,1,2,0}"} : (tensor<1x8x49x64xf32>) -> tensor<1x49x8x64xf32>
    %2 = mhlo.copy %1 : tensor<1x49x8x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x8x64xf32>) -> tensor<1x49x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x49x512xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x49x512xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x49x512xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x49x512xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
    return %12 : tensor<49x512xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<640xf32>, %arg1: tensor<640xf32>, %arg2: tensor<640xf32>, %arg3: tensor<196x640xf32>, %arg4: tensor<640xf32>) -> tensor<8x196x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x640xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x640xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x640xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x640xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:8, 16:80] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,196,64]{3,1,2,0}"} : (tensor<1x196x8x64xf32>) -> tensor<1x8x196x64xf32>
    %11 = mhlo.copy %10 : tensor<1x8x196x64xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x196x64xf32>) -> tensor<8x196x64xf32>
    return %12 : tensor<8x196x64xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<1x8x49x196xf32>, %arg1: tensor<1x8x49xf32>) -> tensor<8x49x196xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x8x49x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x196xf32>) -> tensor<8x49x196xf32>
    return %3 : tensor<8x49x196xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<1x8x49xf32>, %arg1: tensor<9604x8x1xf32>, %arg2: tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<8x49x196xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
    %4 = mhlo.copy %3 : tensor<8x49x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x49x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x8x49xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x8x49xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x8x49x196xf32>
    %12 = stablehlo.exponential %11 : tensor<1x8x49x196xf32>
    return %12 : tensor<1x8x49x196xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<9604x8x1xf32>, %arg1: tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8x49x196xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
    %4 = mhlo.copy %3 : tensor<8x49x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<8x49x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    return %6 : tensor<1x8x49x196xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<8x196xf32>, %arg1: tensor<49x196xi32>) -> tensor<9604x8x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<49x196xi32>, tensor<49x196xi32>) -> tensor<49x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<49x196xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<49x196xi1>, tensor<49x196xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[9604,1]{0,1}"} : (tensor<49x196xi32>) -> tensor<9604x1xi32>
    %6 = mhlo.copy %5 : tensor<9604x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x196xf32>, tensor<9604x1xi32>) -> tensor<9604x8x1xf32>
    return %7 : tensor<9604x8x1xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<640xf32>, %arg1: tensor<640xf32>, %arg2: tensor<640xf32>, %arg3: tensor<196x640xf32>, %arg4: tensor<640xf32>) -> tensor<8x16x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x640xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x640xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x640xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x640xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:8, 0:16] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,196]{2,1,3,0}"} : (tensor<1x196x8x16xf32>) -> tensor<1x8x16x196xf32>
    %11 = mhlo.copy %10 : tensor<1x8x16x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x196xf32>) -> tensor<8x16x196xf32>
    return %12 : tensor<8x16x196xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<640xf32>) -> tensor<640xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<640xf32>
    %2 = stablehlo.rsqrt %1 : tensor<640xf32>
    return %2 : tensor<640xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<1x196x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<196x128xf32>, %arg5: tensor<128xf32>) -> tensor<196x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<196x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
    %9 = stablehlo.add %arg0, %8 : tensor<1x196x128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    return %10 : tensor<196x128xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<49x128xf32>, %arg4: tensor<128xf32>) -> tensor<8x49x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<49x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<49x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<49x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<49x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x128xf32>) -> tensor<1x49x8x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,49,16]{3,1,2,0}"} : (tensor<1x49x8x16xf32>) -> tensor<1x8x49x16xf32>
    %10 = mhlo.copy %9 : tensor<1x8x49x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x16xf32>) -> tensor<8x49x16xf32>
    return %11 : tensor<8x49x16xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<1x196x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<196x128xf32>, %arg5: tensor<128xf32>) -> tensor<49x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<196x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
    %9 = stablehlo.add %arg0, %8 : tensor<1x196x128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<1x14x14x128xf32>
    %11 = stablehlo.slice %10 [0:1, 0:14:2, 0:14, 0:128] : (tensor<1x14x14x128xf32>) -> tensor<1x7x14x128xf32>
    %12 = stablehlo.slice %11 [0:1, 0:7, 0:14:2, 0:128] : (tensor<1x7x14x128xf32>) -> tensor<1x7x7x128xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x128xf32>) -> tensor<49x128xf32>
    return %13 : tensor<49x128xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x196x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x196x256xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x196x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x196x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
    return %17 : tensor<196x256xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<1x196x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<196x128xf32>, %arg5: tensor<128xf32>) -> tensor<1x196x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<196x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
    %9 = stablehlo.add %arg0, %8 : tensor<1x196x128xf32>
    return %9 : tensor<1x196x128xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<4x196x32xf32>) -> tensor<196x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
    %2 = mhlo.copy %1 : tensor<1x196x4x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x196x128xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x196x128xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x196x128xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x196x128xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    return %12 : tensor<196x128xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x196x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
    %11 = mhlo.copy %10 : tensor<1x4x196x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
    return %12 : tensor<4x196x32xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<1x4x196xf32>) -> tensor<4x196x196xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x4x196x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
    return %3 : tensor<4x196x196xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<1x4x196xf32>, %arg1: tensor<38416x4x1xf32>, %arg2: tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<4x196x196xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
    %4 = mhlo.copy %3 : tensor<4x196x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<4x196x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x4x196xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x4x196x196xf32>
    %12 = stablehlo.exponential %11 : tensor<1x4x196x196xf32>
    return %12 : tensor<1x4x196x196xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<38416x4x1xf32>, %arg1: tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<4x196x196xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
    %4 = mhlo.copy %3 : tensor<4x196x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<4x196x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    return %6 : tensor<1x4x196x196xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<4x196xf32>, %arg1: tensor<196x196xi32>) -> tensor<38416x4x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<196x196xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<196x196xi1>, tensor<196x196xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
    %6 = mhlo.copy %5 : tensor<38416x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
    return %7 : tensor<38416x4x1xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x16x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
    %11 = mhlo.copy %10 : tensor<1x4x16x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    return %12 : tensor<4x16x196xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x196x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
    %11 = mhlo.copy %10 : tensor<1x4x196x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
    return %12 : tensor<4x196x16xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<1x196x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<196x128xf32>, %arg5: tensor<128xf32>) -> tensor<1x196x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<196x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
    %9 = stablehlo.add %arg0, %8 : tensor<1x196x128xf32>
    return %9 : tensor<1x196x128xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<196x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %11 = stablehlo.add %8, %10 : tensor<1x196x256xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %13 = stablehlo.clamp %9, %11, %12 : tensor<1x196x256xf32>
    %14 = stablehlo.multiply %8, %13 : tensor<1x196x256xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x196x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
    return %17 : tensor<196x256xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<196x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<1x14x14x128xf32>, %arg9: tensor<128xf32>) -> tensor<1x196x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<1x14x14x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x14x14x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x14x14x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x14x14x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
    %9 = mhlo.copy %8 : tensor<1x128x14x14xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %11 = stablehlo.transpose %10, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
    %12 = mhlo.copy %11 : tensor<1x196x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %14 = stablehlo.subtract %arg3, %13 : tensor<196x128xf32>
    %15 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<196x128xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<196x128xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
    %20 = stablehlo.add %18, %19 : tensor<196x128xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
    %22 = stablehlo.add %12, %21 : tensor<1x196x128xf32>
    return %22 : tensor<1x196x128xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<4x196x32xf32>) -> tensor<196x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
    %2 = mhlo.copy %1 : tensor<1x196x4x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %6 = stablehlo.add %3, %5 : tensor<1x196x128xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %8 = stablehlo.clamp %4, %6, %7 : tensor<1x196x128xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<1x196x128xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x196x128xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    return %12 : tensor<196x128xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x196x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
    %11 = mhlo.copy %10 : tensor<1x4x196x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
    return %12 : tensor<4x196x32xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<1x4x196xf32>) -> tensor<4x196x196xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<1x4x196x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
    return %3 : tensor<4x196x196xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<1x4x196xf32>, %arg1: tensor<38416x4x1xf32>, %arg2: tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<4x196x196xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
    %4 = mhlo.copy %3 : tensor<4x196x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<4x196x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
    %8 = stablehlo.maximum %7, %arg0 : tensor<1x4x196xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
    %11 = stablehlo.subtract %6, %10 : tensor<1x4x196x196xf32>
    %12 = stablehlo.exponential %11 : tensor<1x4x196x196xf32>
    return %12 : tensor<1x4x196x196xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<38416x4x1xf32>, %arg1: tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32> {
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<4x196x196xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
    %4 = mhlo.copy %3 : tensor<4x196x196xf32>
    %5 = stablehlo.add %1, %4 : tensor<4x196x196xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    return %6 : tensor<1x4x196x196xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<4x196xf32>, %arg1: tensor<196x196xi32>) -> tensor<38416x4x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
    %c_0 = stablehlo.constant dense<196> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<196x196xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<196x196xi1>, tensor<196x196xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
    %6 = mhlo.copy %5 : tensor<38416x1xi32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
    return %7 : tensor<38416x4x1xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x16x196xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
    %11 = mhlo.copy %10 : tensor<1x4x16x196xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
    return %12 : tensor<4x16x196xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<196x256xf32>, %arg4: tensor<256xf32>) -> tensor<4x196x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<196x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<196x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<196x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<196x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
    %9 = stablehlo.slice %8 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
    %11 = mhlo.copy %10 : tensor<1x4x196x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
    return %12 : tensor<4x196x16xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1x14x14x128xf32>, %arg4: tensor<128xf32>) -> tensor<196x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x14x14x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x14x14x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x14x14x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x14x14x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
    %9 = mhlo.copy %8 : tensor<1x128x14x14xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
    %11 = stablehlo.transpose %10, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
    %12 = mhlo.copy %11 : tensor<1x196x128xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    return %13 : tensor<196x128xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x64x128xf32>
    return %1 : tensor<3x3x64x128xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1x28x28x64xf32>, %arg4: tensor<64xf32>) -> tensor<1x28x28x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x28x28x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x28x28x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x28x28x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x28x28x64xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,28,28]{1,3,2,0}"} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
    %9 = mhlo.copy %8 : tensor<1x64x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %12 = stablehlo.add %9, %11 : tensor<1x64x28x28xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %14 = stablehlo.clamp %10, %12, %13 : tensor<1x64x28x28xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<1x64x28x28xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x64x28x28xf32>
    %18 = stablehlo.transpose %17, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,28,28,64]{2,1,3,0}"} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
    %19 = mhlo.copy %18 : tensor<1x28x28x64xf32>
    return %19 : tensor<1x28x28x64xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x32x64xf32>
    return %1 : tensor<3x3x32x64xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<1x56x56x32xf32>, %arg4: tensor<32xf32>) -> tensor<1x56x56x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x56x56x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x56x56x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x56x56x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x56x56x32xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,56,56]{1,3,2,0}"} : (tensor<1x56x56x32xf32>) -> tensor<1x32x56x56xf32>
    %9 = mhlo.copy %8 : tensor<1x32x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %12 = stablehlo.add %9, %11 : tensor<1x32x56x56xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %14 = stablehlo.clamp %10, %12, %13 : tensor<1x32x56x56xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<1x32x56x56xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x32x56x56xf32>
    %18 = stablehlo.transpose %17, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,56,56,32]{2,1,3,0}"} : (tensor<1x32x56x56xf32>) -> tensor<1x56x56x32xf32>
    %19 = mhlo.copy %18 : tensor<1x56x56x32xf32>
    return %19 : tensor<1x56x56x32xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,16,32]{1,0,2,3}"} : (tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32>
    %1 = mhlo.copy %0 : tensor<3x3x16x32xf32>
    return %1 : tensor<3x3x16x32xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1x112x112x16xf32>, %arg4: tensor<16xf32>) -> tensor<1x112x112x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x112x112x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x112x112x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x112x112x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x112x112x16xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,112,112]{1,3,2,0}"} : (tensor<1x112x112x16xf32>) -> tensor<1x16x112x112xf32>
    %9 = mhlo.copy %8 : tensor<1x16x112x112xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %12 = stablehlo.add %9, %11 : tensor<1x16x112x112xf32>
    %cst_1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %14 = stablehlo.clamp %10, %12, %13 : tensor<1x16x112x112xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<1x16x112x112xf32>
    %cst_2 = stablehlo.constant dense<0.166666672> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x16x112x112xf32>
    %18 = stablehlo.transpose %17, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,112,112,16]{2,1,3,0}"} : (tensor<1x16x112x112xf32>) -> tensor<1x112x112x16xf32>
    %19 = mhlo.copy %18 : tensor<1x112x112x16xf32>
    return %19 : tensor<1x112x112x16xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<16xf32>
    %2 = stablehlo.rsqrt %1 : tensor<16xf32>
    return %2 : tensor<16xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,16]{1,0,2,3}"} : (tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32>
    %1 = mhlo.copy %0 : tensor<3x3x3x16xf32>
    return %1 : tensor<3x3x3x16xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<1x224x224x3xf32>
    return %1 : tensor<1x224x224x3xf32>
  }
  func.func private @region_0.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<f32>) -> tensor<1x4x196x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    return %0 : tensor<1x4x196x7xf32>
  }
  func.func private @region_0.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<1x4x196x7xf32>, %arg1: tensor<f32>) -> tensor<1x4x196xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    return %0 : tensor<1x4x196xf32>
  }
  func.func private @region_2.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<1x4x196x196xf32>, %arg1: tensor<f32>) -> tensor<1x4x196x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    return %0 : tensor<1x4x196x7xf32>
  }
  func.func private @region_2.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<1x4x196x7xf32>, %arg1: tensor<f32>) -> tensor<1x4x196xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    return %0 : tensor<1x4x196xf32>
  }
  func.func private @region_4.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<1x8x49x196xf32>, %arg1: tensor<f32>) -> tensor<1x8x49x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49x7xf32>
    return %0 : tensor<1x8x49x7xf32>
  }
  func.func private @region_4.15.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<1x8x49x7xf32>, %arg1: tensor<f32>) -> tensor<1x8x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x49x7xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    return %0 : tensor<1x8x49xf32>
  }
  func.func private @region_6.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_6.19.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_7.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_7.20.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_8.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_8.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_9.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_9.24.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_10.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_10.25.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_11.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<1x6x49x49xf32>, %arg1: tensor<f32>) -> tensor<1x6x49x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    return %0 : tensor<1x6x49x2xf32>
  }
  func.func private @region_11.26.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<1x6x49x2xf32>, %arg1: tensor<f32>) -> tensor<1x6x49xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    return %0 : tensor<1x6x49xf32>
  }
  func.func private @region_12.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<1x16x16x49xf32>, %arg1: tensor<f32>) -> tensor<1x16x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    return %0 : tensor<1x16x16x2xf32>
  }
  func.func private @region_12.27.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<1x16x16x2xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_13.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<1x16x16x49xf32>, %arg1: tensor<f32>) -> tensor<1x16x16x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    return %0 : tensor<1x16x16x2xf32>
  }
  func.func private @region_13.28.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<1x16x16x2xf32>, %arg1: tensor<f32>) -> tensor<1x16x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    return %0 : tensor<1x16x16xf32>
  }
  func.func private @region_15.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<f32>) -> tensor<1x8x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %0 : tensor<1x8x16xf32>
  }
  func.func private @region_17.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<f32>) -> tensor<1x8x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %0 : tensor<1x8x16xf32>
  }
  func.func private @region_19.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<f32>) -> tensor<1x8x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %0 : tensor<1x8x16xf32>
  }
  func.func private @region_21.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<1x8x16x16xf32>, %arg1: tensor<f32>) -> tensor<1x8x16xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    return %0 : tensor<1x8x16xf32>
  }
  func.func @main(%arg0: tensor<16x3x3x3xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<32x16x3x3xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<64x32x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<128x64x3x3xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<4x196xf32>, %arg13: tensor<256x128xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<128x128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<256x128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<128x256xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<4x196xf32>, %arg26: tensor<256x128xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<256x128xf32>, %arg33: tensor<256xf32>, %arg34: tensor<256xf32>, %arg35: tensor<128x256xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<8x196xf32>, %arg39: tensor<640x128xf32>, %arg40: tensor<640xf32>, %arg41: tensor<640xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<256x512xf32>, %arg46: tensor<256xf32>, %arg47: tensor<256xf32>, %arg48: tensor<512x256xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<256x512xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<6x49xf32>, %arg55: tensor<384x256xf32>, %arg56: tensor<384xf32>, %arg57: tensor<384xf32>, %arg58: tensor<256x192xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<512x256xf32>, %arg62: tensor<512xf32>, %arg63: tensor<512xf32>, %arg64: tensor<256x512xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<6x49xf32>, %arg68: tensor<384x256xf32>, %arg69: tensor<384xf32>, %arg70: tensor<384xf32>, %arg71: tensor<256x192xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<512x256xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<256x512xf32>, %arg78: tensor<256xf32>, %arg79: tensor<256xf32>, %arg80: tensor<6x49xf32>, %arg81: tensor<384x256xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<256x192xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<512x256xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<256x512xf32>, %arg91: tensor<256xf32>, %arg92: tensor<256xf32>, %arg93: tensor<16x49xf32>, %arg94: tensor<1280x256xf32>, %arg95: tensor<1280xf32>, %arg96: tensor<1280xf32>, %arg97: tensor<256x256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<256xf32>, %arg100: tensor<384x1024xf32>, %arg101: tensor<384xf32>, %arg102: tensor<384xf32>, %arg103: tensor<768x384xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<384x768xf32>, %arg107: tensor<384xf32>, %arg108: tensor<384xf32>, %arg109: tensor<8x16xf32>, %arg110: tensor<512x384xf32>, %arg111: tensor<512xf32>, %arg112: tensor<512xf32>, %arg113: tensor<384x256xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<768x384xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<384x768xf32>, %arg120: tensor<384xf32>, %arg121: tensor<384xf32>, %arg122: tensor<8x16xf32>, %arg123: tensor<512x384xf32>, %arg124: tensor<512xf32>, %arg125: tensor<512xf32>, %arg126: tensor<384x256xf32>, %arg127: tensor<384xf32>, %arg128: tensor<384xf32>, %arg129: tensor<768x384xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<384x768xf32>, %arg133: tensor<384xf32>, %arg134: tensor<384xf32>, %arg135: tensor<8x16xf32>, %arg136: tensor<512x384xf32>, %arg137: tensor<512xf32>, %arg138: tensor<512xf32>, %arg139: tensor<384x256xf32>, %arg140: tensor<384xf32>, %arg141: tensor<384xf32>, %arg142: tensor<768x384xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<384x768xf32>, %arg146: tensor<384xf32>, %arg147: tensor<384xf32>, %arg148: tensor<8x16xf32>, %arg149: tensor<512x384xf32>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<384x256xf32>, %arg153: tensor<384xf32>, %arg154: tensor<384xf32>, %arg155: tensor<768x384xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<384x768xf32>, %arg159: tensor<384xf32>, %arg160: tensor<384xf32>, %arg161: tensor<384xf32>, %arg162: tensor<384xf32>, %arg163: tensor<1000x384xf32>, %arg164: tensor<1000xf32>, %arg165: tensor<16xf32>, %arg166: tensor<16xf32>, %arg167: tensor<32xf32>, %arg168: tensor<32xf32>, %arg169: tensor<64xf32>, %arg170: tensor<64xf32>, %arg171: tensor<128xf32>, %arg172: tensor<128xf32>, %arg173: tensor<196x196xi32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<128xf32>, %arg177: tensor<128xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256xf32>, %arg180: tensor<128xf32>, %arg181: tensor<128xf32>, %arg182: tensor<196x196xi32>, %arg183: tensor<256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<128xf32>, %arg190: tensor<128xf32>, %arg191: tensor<49x196xi32>, %arg192: tensor<640xf32>, %arg193: tensor<640xf32>, %arg194: tensor<128xf32>, %arg195: tensor<128xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256xf32>, %arg198: tensor<512xf32>, %arg199: tensor<512xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<49x49xi32>, %arg203: tensor<384xf32>, %arg204: tensor<384xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<49x49xi32>, %arg212: tensor<384xf32>, %arg213: tensor<384xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<512xf32>, %arg217: tensor<512xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<49x49xi32>, %arg221: tensor<384xf32>, %arg222: tensor<384xf32>, %arg223: tensor<256xf32>, %arg224: tensor<256xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<256xf32>, %arg228: tensor<256xf32>, %arg229: tensor<16x49xi32>, %arg230: tensor<1280xf32>, %arg231: tensor<1280xf32>, %arg232: tensor<256xf32>, %arg233: tensor<256xf32>, %arg234: tensor<384xf32>, %arg235: tensor<384xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<384xf32>, %arg239: tensor<384xf32>, %arg240: tensor<16x16xi32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<384xf32>, %arg244: tensor<384xf32>, %arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<384xf32>, %arg248: tensor<384xf32>, %arg249: tensor<16x16xi32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<384xf32>, %arg253: tensor<384xf32>, %arg254: tensor<768xf32>, %arg255: tensor<768xf32>, %arg256: tensor<384xf32>, %arg257: tensor<384xf32>, %arg258: tensor<16x16xi32>, %arg259: tensor<512xf32>, %arg260: tensor<512xf32>, %arg261: tensor<384xf32>, %arg262: tensor<384xf32>, %arg263: tensor<768xf32>, %arg264: tensor<768xf32>, %arg265: tensor<384xf32>, %arg266: tensor<384xf32>, %arg267: tensor<16x16xi32>, %arg268: tensor<512xf32>, %arg269: tensor<512xf32>, %arg270: tensor<384xf32>, %arg271: tensor<384xf32>, %arg272: tensor<768xf32>, %arg273: tensor<768xf32>, %arg274: tensor<384xf32>, %arg275: tensor<384xf32>, %arg276: tensor<384xf32>, %arg277: tensor<384xf32>, %arg278: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "mhlo.fusion"(%arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %1 = "mhlo.fusion"(%arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %2 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %3 = "mhlo.fusion"(%arg93, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x49xf32>, %arg280: tensor<16x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x49xi1>, tensor<16x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[784,1]{0,1}"} : (tensor<16x49xi32>) -> tensor<784x1xi32>
      %295 = mhlo.copy %294 : tensor<784x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 1>}> : (tensor<16x49xf32>, tensor<784x1xi32>) -> tensor<784x16x1xf32>
      mhlo.return %296 : tensor<784x16x1xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49xf32>, tensor<16x49xi32>) -> tensor<784x16x1xf32>
    %4 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %5 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %6 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %7 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %8 = "mhlo.fusion"(%arg38, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x196xf32>, %arg280: tensor<49x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x196xi32>, tensor<49x196xi32>) -> tensor<49x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x196xi1>, tensor<49x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[9604,1]{0,1}"} : (tensor<49x196xi32>) -> tensor<9604x1xi32>
      %295 = mhlo.copy %294 : tensor<9604x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x196xf32>, tensor<9604x1xi32>) -> tensor<9604x8x1xf32>
      mhlo.return %296 : tensor<9604x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x196xf32>, tensor<49x196xi32>) -> tensor<9604x8x1xf32>
    %9 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %10 = "mhlo.fusion"(%arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = "mhlo.fusion"(%arg12, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196xf32>, %arg280: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<196x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<196x196xi1>, tensor<196x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %295 = mhlo.copy %294 : tensor<38416x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      mhlo.return %296 : tensor<38416x4x1xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<38416x4x1xf32>
    %12 = "mhlo.fusion"(%arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %14 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<64xf32>
      %291 = stablehlo.rsqrt %290 : tensor<64xf32>
      mhlo.return %291 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %15 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<32xf32>
      %291 = stablehlo.rsqrt %290 : tensor<32xf32>
      mhlo.return %291 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %16 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<16xf32>
      %291 = stablehlo.rsqrt %290 : tensor<16xf32>
      mhlo.return %291 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %17 = "mhlo.fusion"(%arg278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x3x224x224xf32>):
      %289 = stablehlo.transpose %arg279, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %290 = mhlo.copy %289 : tensor<1x224x224x3xf32>
      mhlo.return %290 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %18 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x3x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,16]{1,0,2,3}"} : (tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32>
      %290 = mhlo.copy %289 : tensor<3x3x3x16xf32>
      mhlo.return %290 : tensor<3x3x3x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32>
    %19 = stablehlo.convolution(%17, %18) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x112x112x16xf32>
    %20 = "mhlo.fusion"(%arg2, %arg1, %16, %19, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16xf32>, %arg280: tensor<16xf32>, %arg281: tensor<16xf32>, %arg282: tensor<1x112x112x16xf32>, %arg283: tensor<16xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x112x112x16xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x112x112x16xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x112x112x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x112x112x16xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,112,112]{1,3,2,0}"} : (tensor<1x112x112x16xf32>) -> tensor<1x16x112x112xf32>
      %298 = mhlo.copy %297 : tensor<1x16x112x112xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x16x112x112xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x16x112x112xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x16x112x112xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x16x112x112xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,112,112,16]{2,1,3,0}"} : (tensor<1x16x112x112xf32>) -> tensor<1x112x112x16xf32>
      %308 = mhlo.copy %307 : tensor<1x112x112x16xf32>
      mhlo.return %308 : tensor<1x112x112x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<1x112x112x16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %21 = "mhlo.fusion"(%arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32x16x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,16,32]{1,0,2,3}"} : (tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32>
      %290 = mhlo.copy %289 : tensor<3x3x16x32xf32>
      mhlo.return %290 : tensor<3x3x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32>
    %22 = stablehlo.convolution(%20, %21) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x112x112x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x56x56x32xf32>
    %23 = "mhlo.fusion"(%arg5, %arg4, %15, %22, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32xf32>, %arg280: tensor<32xf32>, %arg281: tensor<32xf32>, %arg282: tensor<1x56x56x32xf32>, %arg283: tensor<32xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x56x56x32xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x56x56x32xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56x32xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x56x56x32xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,56,56]{1,3,2,0}"} : (tensor<1x56x56x32xf32>) -> tensor<1x32x56x56xf32>
      %298 = mhlo.copy %297 : tensor<1x32x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x32x56x56xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x32x56x56xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x32x56x56xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x32x56x56xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,56,56,32]{2,1,3,0}"} : (tensor<1x32x56x56xf32>) -> tensor<1x56x56x32xf32>
      %308 = mhlo.copy %307 : tensor<1x56x56x32xf32>
      mhlo.return %308 : tensor<1x56x56x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x56x56x32xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %24 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64x32x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %290 = mhlo.copy %289 : tensor<3x3x32x64xf32>
      mhlo.return %290 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x32xf32>, tensor<3x3x32x64xf32>) -> tensor<1x28x28x64xf32>
    %26 = "mhlo.fusion"(%arg8, %arg7, %14, %25, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64xf32>, %arg280: tensor<64xf32>, %arg281: tensor<64xf32>, %arg282: tensor<1x28x28x64xf32>, %arg283: tensor<64xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x28x28x64xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x28x28x64xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28x64xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x28x28x64xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,28,28]{1,3,2,0}"} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
      %298 = mhlo.copy %297 : tensor<1x64x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x64x28x28xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x64x28x28xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x64x28x28xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x64x28x28xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,28,28,64]{2,1,3,0}"} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
      %308 = mhlo.copy %307 : tensor<1x28x28x64xf32>
      mhlo.return %308 : tensor<1x28x28x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x28x28x64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %27 = "mhlo.fusion"(%arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128x64x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32>
      %290 = mhlo.copy %289 : tensor<3x3x64x128xf32>
      mhlo.return %290 : tensor<3x3x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32>
    %28 = stablehlo.convolution(%26, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x14x14x128xf32>
    %29 = "mhlo.fusion"(%arg11, %arg10, %13, %28, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<1x14x14x128xf32>, %arg283: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x14x14x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x14x14x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14x128xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
      %298 = mhlo.copy %297 : tensor<1x128x14x14xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
      %301 = mhlo.copy %300 : tensor<1x196x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %302 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x14x14x128xf32>, tensor<128xf32>) -> tensor<196x128xf32>
    %30 = stablehlo.dot_general %29, %arg13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %31 = "mhlo.fusion"(%arg15, %arg14, %12, %30, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
      mhlo.return %301 : tensor<4x196x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x16xf32>
    %32 = "mhlo.fusion"(%arg15, %arg14, %12, %30, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x4x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
      mhlo.return %301 : tensor<4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x16x196xf32>
    %33 = stablehlo.dot_general %31, %32, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x16xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %34 = "mhlo.fusion"(%11, %33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<38416x4x1xf32>, %arg280: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      mhlo.return %295 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
      mhlo.return %289 : tensor<1x4x196x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    %36 = "mhlo.fusion"(%35, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %37 = "mhlo.fusion"(%36, %11, %33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196xf32>, %arg280: tensor<38416x4x1xf32>, %arg281: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x4x196xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x4x196x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x4x196x196xf32>
      mhlo.return %301 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196xf32>, tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = "mhlo.fusion"(%37, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %39 = "mhlo.fusion"(%37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<1x4x196xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x4x196x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
      mhlo.return %292 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<1x4x196xf32>) -> tensor<4x196x196xf32>
    %40 = "mhlo.fusion"(%arg15, %arg14, %12, %30, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
      mhlo.return %301 : tensor<4x196x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x32xf32>
    %41 = stablehlo.dot_general %39, %40, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x196x32xf32>) -> tensor<4x196x32xf32>
    %42 = "mhlo.fusion"(%41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
      %291 = mhlo.copy %290 : tensor<1x196x4x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x196x128xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x196x128xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x196x128xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x196x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %301 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x32xf32>) -> tensor<196x128xf32>
    %43 = stablehlo.dot_general %42, %arg16, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %44 = "mhlo.fusion"(%arg18, %arg17, %10, %43, %arg176, %arg11, %arg10, %13, %28, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<128xf32>, %arg287: tensor<1x14x14x128xf32>, %arg288: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<1x14x14x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x14x14x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14x128xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
      %298 = mhlo.copy %297 : tensor<1x128x14x14xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
      %301 = mhlo.copy %300 : tensor<1x196x128xf32>
      %302 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %303 = stablehlo.subtract %arg282, %302 : tensor<196x128xf32>
      %304 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<196x128xf32>
      %306 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<196x128xf32>
      %308 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %309 = stablehlo.add %307, %308 : tensor<196x128xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %311 = stablehlo.add %301, %310 : tensor<1x196x128xf32>
      mhlo.return %311 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x14x14x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %45 = "mhlo.fusion"(%arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %46 = "mhlo.fusion"(%arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %47 = mhlo.bitcast %44 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %48 = stablehlo.dot_general %47, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %49 = "mhlo.fusion"(%arg21, %arg20, %46, %48, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x196x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x196x256xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x196x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x196x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
      mhlo.return %306 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<196x256xf32>
    %50 = stablehlo.dot_general %49, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %51 = "mhlo.fusion"(%44, %arg24, %arg23, %45, %50, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      mhlo.return %298 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %52 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %53 = "mhlo.fusion"(%arg25, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196xf32>, %arg280: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<196x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<196x196xi1>, tensor<196x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %295 = mhlo.copy %294 : tensor<38416x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      mhlo.return %296 : tensor<38416x4x1xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<38416x4x1xf32>
    %54 = "mhlo.fusion"(%arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %55 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %56 = stablehlo.dot_general %55, %arg26, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %57 = "mhlo.fusion"(%arg28, %arg27, %54, %56, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
      mhlo.return %301 : tensor<4x196x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x16xf32>
    %58 = "mhlo.fusion"(%arg28, %arg27, %54, %56, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x4x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
      mhlo.return %301 : tensor<4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x16x196xf32>
    %59 = stablehlo.dot_general %57, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x16xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %60 = "mhlo.fusion"(%53, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<38416x4x1xf32>, %arg280: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      mhlo.return %295 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %61 = "mhlo.fusion"(%60, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
      mhlo.return %289 : tensor<1x4x196x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    %62 = "mhlo.fusion"(%61, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %63 = "mhlo.fusion"(%62, %53, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196xf32>, %arg280: tensor<38416x4x1xf32>, %arg281: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x4x196xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x4x196x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x4x196x196xf32>
      mhlo.return %301 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196xf32>, tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %64 = "mhlo.fusion"(%63, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %65 = "mhlo.fusion"(%63, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<1x4x196xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x4x196x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
      mhlo.return %292 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<1x4x196xf32>) -> tensor<4x196x196xf32>
    %66 = "mhlo.fusion"(%arg28, %arg27, %54, %56, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
      mhlo.return %301 : tensor<4x196x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x32xf32>
    %67 = stablehlo.dot_general %65, %66, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x196x32xf32>) -> tensor<4x196x32xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
      %291 = mhlo.copy %290 : tensor<1x196x4x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x196x128xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x196x128xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x196x128xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x196x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %301 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x32xf32>) -> tensor<196x128xf32>
    %69 = stablehlo.dot_general %68, %arg29, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %70 = "mhlo.fusion"(%51, %arg31, %arg30, %52, %69, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      mhlo.return %298 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %71 = "mhlo.fusion"(%arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %72 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %73 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %74 = stablehlo.dot_general %73, %arg32, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %75 = "mhlo.fusion"(%arg34, %arg33, %72, %74, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x196x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x196x256xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x196x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x196x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
      mhlo.return %306 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<196x256xf32>
    %76 = stablehlo.dot_general %75, %arg35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %77 = "mhlo.fusion"(%70, %arg37, %arg36, %71, %76, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<1x14x14x128xf32>
      %300 = stablehlo.slice %299 [0:1, 0:14:2, 0:14, 0:128] : (tensor<1x14x14x128xf32>) -> tensor<1x7x14x128xf32>
      %301 = stablehlo.slice %300 [0:1, 0:7, 0:14:2, 0:128] : (tensor<1x7x14x128xf32>) -> tensor<1x7x7x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x128xf32>) -> tensor<49x128xf32>
      mhlo.return %302 : tensor<49x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<49x128xf32>
    %78 = stablehlo.dot_general %77, %arg42, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x128xf32>, tensor<128x128xf32>) -> tensor<49x128xf32>
    %79 = "mhlo.fusion"(%arg44, %arg43, %9, %78, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<49x128xf32>, %arg283: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x128xf32>) -> tensor<1x49x8x16xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,49,16]{3,1,2,0}"} : (tensor<1x49x8x16xf32>) -> tensor<1x8x49x16xf32>
      %299 = mhlo.copy %298 : tensor<1x8x49x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x16xf32>) -> tensor<8x49x16xf32>
      mhlo.return %300 : tensor<8x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<49x128xf32>, tensor<128xf32>) -> tensor<8x49x16xf32>
    %80 = "mhlo.fusion"(%arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<640xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<640xf32>
      %291 = stablehlo.rsqrt %290 : tensor<640xf32>
      mhlo.return %291 : tensor<640xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>) -> tensor<640xf32>
    %81 = "mhlo.fusion"(%70, %arg37, %arg36, %71, %76, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %299 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<196x128xf32>
    %82 = stablehlo.dot_general %81, %arg39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<640x128xf32>) -> tensor<196x640xf32>
    %83 = "mhlo.fusion"(%arg41, %arg40, %80, %82, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<196x640xf32>, %arg283: tensor<640xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x640xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x640xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x640xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x640xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:8, 0:16] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,196]{2,1,3,0}"} : (tensor<1x196x8x16xf32>) -> tensor<1x8x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x196xf32>) -> tensor<8x16x196xf32>
      mhlo.return %301 : tensor<8x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<196x640xf32>, tensor<640xf32>) -> tensor<8x16x196xf32>
    %84 = stablehlo.dot_general %79, %83, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x49x16xf32>, tensor<8x16x196xf32>) -> tensor<8x49x196xf32>
    %85 = "mhlo.fusion"(%8, %84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<9604x8x1xf32>, %arg280: tensor<8x49x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x49x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
      %293 = mhlo.copy %292 : tensor<8x49x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x49x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
      mhlo.return %295 : tensor<1x8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<9604x8x1xf32>, tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49x7xf32>
      mhlo.return %289 : tensor<1x8x49x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49x7xf32>
    %87 = "mhlo.fusion"(%86, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x49x7xf32>, tensor<f32>) -> tensor<1x8x49xf32>
      mhlo.return %289 : tensor<1x8x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x7xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    %88 = "mhlo.fusion"(%87, %8, %84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49xf32>, %arg280: tensor<9604x8x1xf32>, %arg281: tensor<8x49x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x49x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
      %293 = mhlo.copy %292 : tensor<8x49x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x49x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x49x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x49x196xf32>
      mhlo.return %301 : tensor<1x8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49xf32>, tensor<9604x8x1xf32>, tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    %89 = "mhlo.fusion"(%88, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49xf32>
      mhlo.return %289 : tensor<1x8x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    %90 = "mhlo.fusion"(%88, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<1x8x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x49x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x196xf32>) -> tensor<8x49x196xf32>
      mhlo.return %292 : tensor<8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<1x8x49xf32>) -> tensor<8x49x196xf32>
    %91 = "mhlo.fusion"(%arg41, %arg40, %80, %82, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<196x640xf32>, %arg283: tensor<640xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x640xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x640xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x640xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x640xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:8, 16:80] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x64xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,196,64]{3,1,2,0}"} : (tensor<1x196x8x64xf32>) -> tensor<1x8x196x64xf32>
      %300 = mhlo.copy %299 : tensor<1x8x196x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x196x64xf32>) -> tensor<8x196x64xf32>
      mhlo.return %301 : tensor<8x196x64xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<196x640xf32>, tensor<640xf32>) -> tensor<8x196x64xf32>
    %92 = stablehlo.dot_general %90, %91, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x49x196xf32>, tensor<8x196x64xf32>) -> tensor<8x49x64xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x49x64xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x64xf32>) -> tensor<1x8x49x64xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,8,64]{3,1,2,0}"} : (tensor<1x8x49x64xf32>) -> tensor<1x49x8x64xf32>
      %291 = mhlo.copy %290 : tensor<1x49x8x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x8x64xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x512xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x512xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %301 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<8x49x64xf32>) -> tensor<49x512xf32>
    %94 = stablehlo.dot_general %93, %arg45, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %95 = "mhlo.fusion"(%arg47, %arg46, %7, %94, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<49x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      mhlo.return %296 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %96 = stablehlo.dot_general %95, %arg48, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %97 = "mhlo.fusion"(%arg50, %arg49, %6, %96, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %98 = stablehlo.dot_general %97, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %99 = "mhlo.fusion"(%arg53, %arg52, %5, %98, %arg200, %arg47, %arg46, %7, %94, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<49x256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<49x256xf32>, %arg288: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %298 = stablehlo.subtract %arg282, %297 : tensor<49x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<49x256xf32>
      %301 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<49x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<49x256xf32>
      %305 = stablehlo.add %296, %304 : tensor<49x256xf32>
      mhlo.return %305 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %100 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %101 = "mhlo.fusion"(%arg54, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %102 = "mhlo.fusion"(%arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %103 = stablehlo.dot_general %99, %arg55, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %104 = "mhlo.fusion"(%arg57, %arg56, %102, %103, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %105 = "mhlo.fusion"(%arg57, %arg56, %102, %103, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %106 = stablehlo.dot_general %104, %105, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %107 = "mhlo.fusion"(%101, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %110 = "mhlo.fusion"(%109, %101, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %111 = "mhlo.fusion"(%110, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %112 = "mhlo.fusion"(%111, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %113 = "mhlo.fusion"(%110, %112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %114 = "mhlo.fusion"(%arg57, %arg56, %102, %103, %arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %115 = stablehlo.dot_general %113, %114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %117 = stablehlo.dot_general %116, %arg58, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %118 = "mhlo.fusion"(%99, %arg60, %arg59, %100, %117, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %119 = "mhlo.fusion"(%arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %120 = "mhlo.fusion"(%arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %121 = stablehlo.dot_general %118, %arg61, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %122 = "mhlo.fusion"(%arg63, %arg62, %120, %121, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %123 = stablehlo.dot_general %122, %arg64, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %124 = "mhlo.fusion"(%118, %arg66, %arg65, %119, %123, %arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %125 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %126 = "mhlo.fusion"(%arg67, %arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %127 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %128 = stablehlo.dot_general %124, %arg68, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %129 = "mhlo.fusion"(%arg70, %arg69, %127, %128, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %130 = "mhlo.fusion"(%arg70, %arg69, %127, %128, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %131 = stablehlo.dot_general %129, %130, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %132 = "mhlo.fusion"(%126, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %134 = "mhlo.fusion"(%133, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %135 = "mhlo.fusion"(%134, %126, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %136 = "mhlo.fusion"(%135, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %137 = "mhlo.fusion"(%136, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %138 = "mhlo.fusion"(%135, %137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %139 = "mhlo.fusion"(%arg70, %arg69, %127, %128, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %140 = stablehlo.dot_general %138, %139, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %141 = "mhlo.fusion"(%140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %142 = stablehlo.dot_general %141, %arg71, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %143 = "mhlo.fusion"(%124, %arg73, %arg72, %125, %142, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %144 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %145 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %146 = stablehlo.dot_general %143, %arg74, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %147 = "mhlo.fusion"(%arg76, %arg75, %145, %146, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %148 = stablehlo.dot_general %147, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %149 = "mhlo.fusion"(%143, %arg79, %arg78, %144, %148, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %150 = "mhlo.fusion"(%arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %151 = "mhlo.fusion"(%arg80, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %152 = "mhlo.fusion"(%arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %153 = stablehlo.dot_general %149, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %154 = "mhlo.fusion"(%arg83, %arg82, %152, %153, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %155 = "mhlo.fusion"(%arg83, %arg82, %152, %153, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %156 = stablehlo.dot_general %154, %155, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %157 = "mhlo.fusion"(%151, %156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %158 = "mhlo.fusion"(%157, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %159 = "mhlo.fusion"(%158, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %160 = "mhlo.fusion"(%159, %151, %156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %161 = "mhlo.fusion"(%160, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %162 = "mhlo.fusion"(%161, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %163 = "mhlo.fusion"(%160, %162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %164 = "mhlo.fusion"(%arg83, %arg82, %152, %153, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %165 = stablehlo.dot_general %163, %164, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %166 = "mhlo.fusion"(%165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %167 = stablehlo.dot_general %166, %arg84, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %168 = "mhlo.fusion"(%149, %arg86, %arg85, %150, %167, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %169 = "mhlo.fusion"(%arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %170 = "mhlo.fusion"(%arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %171 = stablehlo.dot_general %168, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %172 = "mhlo.fusion"(%arg89, %arg88, %170, %171, %arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %173 = stablehlo.dot_general %172, %arg90, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %174 = "mhlo.fusion"(%168, %arg92, %arg91, %169, %173, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x256xf32>) -> tensor<1x7x7x256xf32>
      %290 = stablehlo.slice %289 [0:1, 0:7:2, 0:7, 0:256] : (tensor<1x7x7x256xf32>) -> tensor<1x4x7x256xf32>
      %291 = stablehlo.slice %290 [0:1, 0:4, 0:7:2, 0:256] : (tensor<1x4x7x256xf32>) -> tensor<1x4x4x256xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x4x256xf32>) -> tensor<16x256xf32>
      mhlo.return %292 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>) -> tensor<16x256xf32>
    %176 = stablehlo.dot_general %175, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<256x256xf32>) -> tensor<16x256xf32>
    %177 = "mhlo.fusion"(%arg99, %arg98, %4, %176, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<16x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x256xf32>) -> tensor<1x16x16x16xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,16]{3,1,2,0}"} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
      %299 = mhlo.copy %298 : tensor<1x16x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x16xf32>) -> tensor<16x16x16xf32>
      mhlo.return %300 : tensor<16x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<256xf32>) -> tensor<16x16x16xf32>
    %178 = "mhlo.fusion"(%arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<1280xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1280xf32>
      mhlo.return %291 : tensor<1280xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>) -> tensor<1280xf32>
    %179 = stablehlo.dot_general %174, %arg94, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<1280x256xf32>) -> tensor<49x1280xf32>
    %180 = "mhlo.fusion"(%arg96, %arg95, %178, %179, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<49x1280xf32>, %arg283: tensor<1280xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x1280xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x1280xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x1280xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x1280xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:16, 0:16] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,49]{2,1,3,0}"} : (tensor<1x49x16x16xf32>) -> tensor<1x16x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x16x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
      mhlo.return %301 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<49x1280xf32>, tensor<1280xf32>) -> tensor<16x16x49xf32>
    %181 = stablehlo.dot_general %177, %180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x16xf32>, tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
    %182 = "mhlo.fusion"(%3, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<784x16x1xf32>, %arg280: tensor<16x16x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<16x16x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
      %293 = mhlo.copy %292 : tensor<16x16x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<16x16x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
      mhlo.return %295 : tensor<1x16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<784x16x1xf32>, tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    %183 = "mhlo.fusion"(%182, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
      mhlo.return %289 : tensor<1x16x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %289 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %185 = "mhlo.fusion"(%184, %3, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16xf32>, %arg280: tensor<784x16x1xf32>, %arg281: tensor<16x16x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<16x16x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
      %293 = mhlo.copy %292 : tensor<16x16x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<16x16x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x16x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x16x16x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x16x16x49xf32>
      mhlo.return %301 : tensor<1x16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<784x16x1xf32>, tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    %186 = "mhlo.fusion"(%185, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
      mhlo.return %289 : tensor<1x16x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    %187 = "mhlo.fusion"(%186, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %289 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %188 = "mhlo.fusion"(%185, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<1x16x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x16x16x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
      mhlo.return %292 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<1x16x16xf32>) -> tensor<16x16x49xf32>
    %189 = "mhlo.fusion"(%arg96, %arg95, %178, %179, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<49x1280xf32>, %arg283: tensor<1280xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x1280xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x1280xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x1280xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x1280xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:16, 16:80] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x64xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,49,64]{3,1,2,0}"} : (tensor<1x49x16x64xf32>) -> tensor<1x16x49x64xf32>
      %300 = mhlo.copy %299 : tensor<1x16x49x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x49x64xf32>) -> tensor<16x49x64xf32>
      mhlo.return %301 : tensor<16x49x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<49x1280xf32>, tensor<1280xf32>) -> tensor<16x49x64xf32>
    %190 = stablehlo.dot_general %188, %189, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x49xf32>, tensor<16x49x64xf32>) -> tensor<16x16x64xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x16x64xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x64xf32>) -> tensor<1x16x16x64xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,64]{3,1,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x16x16x64xf32>
      %291 = mhlo.copy %290 : tensor<1x16x16x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x64xf32>) -> tensor<1x16x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x1024xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x1024xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x1024xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x1024xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x1024xf32>) -> tensor<16x1024xf32>
      mhlo.return %301 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x64xf32>) -> tensor<16x1024xf32>
    %192 = stablehlo.dot_general %191, %arg100, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x1024xf32>, tensor<384x1024xf32>) -> tensor<16x384xf32>
    %193 = "mhlo.fusion"(%arg102, %arg101, %2, %192, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<16x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      mhlo.return %296 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %194 = stablehlo.dot_general %193, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %195 = "mhlo.fusion"(%arg105, %arg104, %1, %194, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %196 = stablehlo.dot_general %195, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %197 = "mhlo.fusion"(%arg108, %arg107, %0, %196, %arg238, %arg102, %arg101, %2, %192, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<16x384xf32>, %arg283: tensor<384xf32>, %arg284: tensor<384xf32>, %arg285: tensor<384xf32>, %arg286: tensor<384xf32>, %arg287: tensor<16x384xf32>, %arg288: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %298 = stablehlo.subtract %arg282, %297 : tensor<16x384xf32>
      %299 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<16x384xf32>
      %301 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<16x384xf32>
      %303 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %304 = stablehlo.add %302, %303 : tensor<16x384xf32>
      %305 = stablehlo.add %296, %304 : tensor<16x384xf32>
      mhlo.return %305 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %198 = "mhlo.fusion"(%arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %199 = "mhlo.fusion"(%arg109, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %200 = "mhlo.fusion"(%arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %201 = stablehlo.dot_general %197, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %202 = "mhlo.fusion"(%arg112, %arg111, %200, %201, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %203 = "mhlo.fusion"(%arg112, %arg111, %200, %201, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %204 = stablehlo.dot_general %202, %203, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %205 = "mhlo.fusion"(%199, %204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %206 = "mhlo.fusion"(%205, %199, %204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %207 = "mhlo.fusion"(%206, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %208 = "mhlo.fusion"(%206, %207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %209 = "mhlo.fusion"(%arg112, %arg111, %200, %201, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %210 = stablehlo.dot_general %208, %209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %212 = stablehlo.dot_general %211, %arg113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %213 = "mhlo.fusion"(%197, %arg115, %arg114, %198, %212, %arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %214 = "mhlo.fusion"(%arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %215 = "mhlo.fusion"(%arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %216 = stablehlo.dot_general %213, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %217 = "mhlo.fusion"(%arg118, %arg117, %215, %216, %arg245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %218 = stablehlo.dot_general %217, %arg119, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %219 = "mhlo.fusion"(%213, %arg121, %arg120, %214, %218, %arg247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %220 = "mhlo.fusion"(%arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %221 = "mhlo.fusion"(%arg122, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %222 = "mhlo.fusion"(%arg251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %223 = stablehlo.dot_general %219, %arg123, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %224 = "mhlo.fusion"(%arg125, %arg124, %222, %223, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %225 = "mhlo.fusion"(%arg125, %arg124, %222, %223, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %226 = stablehlo.dot_general %224, %225, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %227 = "mhlo.fusion"(%221, %226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %228 = "mhlo.fusion"(%227, %221, %226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %229 = "mhlo.fusion"(%228, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %230 = "mhlo.fusion"(%228, %229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %231 = "mhlo.fusion"(%arg125, %arg124, %222, %223, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %232 = stablehlo.dot_general %230, %231, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %234 = stablehlo.dot_general %233, %arg126, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %235 = "mhlo.fusion"(%219, %arg128, %arg127, %220, %234, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %236 = "mhlo.fusion"(%arg257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %237 = "mhlo.fusion"(%arg255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %238 = stablehlo.dot_general %235, %arg129, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %239 = "mhlo.fusion"(%arg131, %arg130, %237, %238, %arg254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %240 = stablehlo.dot_general %239, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %241 = "mhlo.fusion"(%235, %arg134, %arg133, %236, %240, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %242 = "mhlo.fusion"(%arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %243 = "mhlo.fusion"(%arg135, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %244 = "mhlo.fusion"(%arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %245 = stablehlo.dot_general %241, %arg136, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %246 = "mhlo.fusion"(%arg138, %arg137, %244, %245, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %247 = "mhlo.fusion"(%arg138, %arg137, %244, %245, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %248 = stablehlo.dot_general %246, %247, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %249 = "mhlo.fusion"(%243, %248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %250 = "mhlo.fusion"(%249, %243, %248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %251 = "mhlo.fusion"(%250, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %252 = "mhlo.fusion"(%250, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %253 = "mhlo.fusion"(%arg138, %arg137, %244, %245, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %254 = stablehlo.dot_general %252, %253, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %256 = stablehlo.dot_general %255, %arg139, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %257 = "mhlo.fusion"(%241, %arg141, %arg140, %242, %256, %arg261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %258 = "mhlo.fusion"(%arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %259 = "mhlo.fusion"(%arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %260 = stablehlo.dot_general %257, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %261 = "mhlo.fusion"(%arg144, %arg143, %259, %260, %arg263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %262 = stablehlo.dot_general %261, %arg145, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %263 = "mhlo.fusion"(%257, %arg147, %arg146, %258, %262, %arg265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %264 = "mhlo.fusion"(%arg271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %265 = "mhlo.fusion"(%arg148, %arg267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %266 = "mhlo.fusion"(%arg269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %267 = stablehlo.dot_general %263, %arg149, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %268 = "mhlo.fusion"(%arg151, %arg150, %266, %267, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %269 = "mhlo.fusion"(%arg151, %arg150, %266, %267, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %270 = stablehlo.dot_general %268, %269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %271 = "mhlo.fusion"(%265, %270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %272 = "mhlo.fusion"(%271, %265, %270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %273 = "mhlo.fusion"(%272, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %274 = "mhlo.fusion"(%272, %273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %275 = "mhlo.fusion"(%arg151, %arg150, %266, %267, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %276 = stablehlo.dot_general %274, %275, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %278 = stablehlo.dot_general %277, %arg152, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %279 = "mhlo.fusion"(%263, %arg154, %arg153, %264, %278, %arg270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %280 = "mhlo.fusion"(%arg275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %281 = "mhlo.fusion"(%arg273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %282 = stablehlo.dot_general %279, %arg155, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %283 = "mhlo.fusion"(%arg157, %arg156, %281, %282, %arg272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %284 = stablehlo.dot_general %283, %arg158, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %285 = "mhlo.fusion"(%arg162, %arg161, %arg277, %arg276, %279, %arg160, %arg159, %280, %284, %arg274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>, %arg285: tensor<384xf32>, %arg286: tensor<384xf32>, %arg287: tensor<16x384xf32>, %arg288: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg283, %296 : tensor<16x384xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.reduce(%297 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<16x384xf32>, tensor<f32>) -> tensor<384xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-02> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<384xf32>
      %301 = stablehlo.subtract %300, %arg282 : tensor<384xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %303 = stablehlo.add %arg281, %302 : tensor<384xf32>
      %304 = stablehlo.rsqrt %303 : tensor<384xf32>
      %305 = stablehlo.multiply %301, %304 : tensor<384xf32>
      %306 = stablehlo.multiply %305, %arg280 : tensor<384xf32>
      %307 = stablehlo.add %306, %arg279 : tensor<384xf32>
      mhlo.return %307 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<384xf32>
    %286 = "mhlo.fusion"(%arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1000x384xf32>):
      %289 = stablehlo.transpose %arg279, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,1000]{0,1}"} : (tensor<1000x384xf32>) -> tensor<384x1000xf32>
      %290 = mhlo.copy %289 : tensor<384x1000xf32>
      mhlo.return %290 : tensor<384x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x384xf32>) -> tensor<384x1000xf32>
    %287 = "mhlo.fusion"(%arg164, %285, %286) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg279: tensor<1000xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x1000xf32>):
      %289 = stablehlo.dot %arg280, %arg281, precision = [DEFAULT, DEFAULT] : (tensor<384xf32>, tensor<384x1000xf32>) -> tensor<1000xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<1000xf32>
      mhlo.return %290 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<384xf32>, tensor<384x1000xf32>) -> tensor<1000xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %288 : tensor<1x1000xf32>
  }
}
