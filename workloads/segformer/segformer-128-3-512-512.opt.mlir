module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.68_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<2097152x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<2097152x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    return %4 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<128x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<32768x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x32x256xf32>) -> tensor<128x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<32768x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<32768x32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
    return %6 : tensor<32768x64xf32>
  }
  func.func private @gemm_fusion_dot_general.71_computation(%arg0: tensor<32xf32>, %arg1: tensor<128x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x256x32xf32>) -> tensor<128x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<128x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<128x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x256x32xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x16384x256xf32>
    return %7 : tensor<128x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.72_computation(%arg0: tensor<128x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<128x256x32xf32>) -> tensor<128x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<128x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x256xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
    return %2 : tensor<128x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.73_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    return %0 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot_general.74_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<2097152x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<2097152x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<2097152x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
    return %4 : tensor<2097152x128xf32>
  }
  func.func private @gemm_fusion_dot_general.75_computation(%arg0: tensor<128x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x128x16384xf32>) -> tensor<128x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x128xf32>) -> tensor<2097152x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
    return %2 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot_general.76_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<2097152x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<2097152x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    return %4 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<128x32x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<32768x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x32x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x32x256xf32>) -> tensor<128x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<32768x32xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<32768x32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
    return %6 : tensor<32768x64xf32>
  }
  func.func private @gemm_fusion_dot_general.79_computation(%arg0: tensor<32xf32>, %arg1: tensor<128x16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x256x32xf32>) -> tensor<128x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<128x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
    %3 = stablehlo.add %2, %arg3 : tensor<128x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x256x32xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.dot_general %1, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16384x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x16384x256xf32>
    return %7 : tensor<128x16384x256xf32>
  }
  func.func private @gemm_fusion_dot_general.80_computation(%arg0: tensor<128x16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<128x256x32xf32>) -> tensor<128x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<128x256x32xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x256xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
    return %2 : tensor<128x16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.81_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    return %0 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot_general.82_computation(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<2097152x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<2097152x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<2097152x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
    return %4 : tensor<2097152x128xf32>
  }
  func.func private @gemm_fusion_dot_general.83_computation(%arg0: tensor<128x128x16384xf32>, %arg1: tensor<32x128xf32>) -> tensor<2097152x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x128x16384xf32>) -> tensor<128x16384x128xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x128xf32>) -> tensor<2097152x128xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
    return %2 : tensor<2097152x32xf32>
  }
  func.func private @gemm_fusion_dot_general.84_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<524288x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<524288x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    return %4 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<128x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<32768x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x64x256xf32>) -> tensor<128x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x64xf32>) -> tensor<32768x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x128xf32>) -> tensor<32768x128xf32>
    return %8 : tensor<32768x128xf32>
  }
  func.func private @gemm_fusion_dot_general.87_computation(%arg0: tensor<256x4096x32xf32>, %arg1: tensor<256x32x256xf32>) -> tensor<256x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x4096x256xf32>
    return %2 : tensor<256x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.88_computation(%arg0: tensor<256x4096x256xf32>, %arg1: tensor<256x256x32xf32>) -> tensor<256x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
    return %0 : tensor<256x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.89_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    return %0 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot_general.90_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<524288x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<524288x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<524288x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
    return %4 : tensor<524288x256xf32>
  }
  func.func private @gemm_fusion_dot_general.91_computation(%arg0: tensor<128x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x256x4096xf32>) -> tensor<128x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<524288x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
    return %2 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot_general.92_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<524288x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<524288x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    return %4 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<128x64x256xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<32768x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x64x256xf32>) -> tensor<128x256x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x64xf32>) -> tensor<32768x64xf32>
    %6 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %8 = stablehlo.dot %5, %7, precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x128xf32>) -> tensor<32768x128xf32>
    return %8 : tensor<32768x128xf32>
  }
  func.func private @gemm_fusion_dot_general.95_computation(%arg0: tensor<256x4096x32xf32>, %arg1: tensor<256x32x256xf32>) -> tensor<256x4096x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x4096x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<256x4096x256xf32>
    return %2 : tensor<256x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.96_computation(%arg0: tensor<256x4096x256xf32>, %arg1: tensor<256x256x32xf32>) -> tensor<256x4096x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
    return %0 : tensor<256x4096x32xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    return %0 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<524288x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<524288x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<524288x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
    return %4 : tensor<524288x256xf32>
  }
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<128x256x4096xf32>, %arg1: tensor<64x256xf32>) -> tensor<524288x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x256x4096xf32>) -> tensor<128x4096x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<524288x256xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
    return %2 : tensor<524288x64xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<131072x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<131072x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    return %4 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<128x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<32768x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x160x256xf32>) -> tensor<128x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x160xf32>) -> tensor<32768x160xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<32768x160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
    return %6 : tensor<32768x320xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<640x1024x32xf32>, %arg1: tensor<640x32x256xf32>) -> tensor<640x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<640x1024x256xf32>
    return %2 : tensor<640x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<640x1024x256xf32>, %arg1: tensor<640x256x32xf32>) -> tensor<640x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
    return %0 : tensor<640x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    return %0 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<131072x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<131072x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<131072x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
    return %4 : tensor<131072x640xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<128x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x640x1024xf32>) -> tensor<128x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x640xf32>) -> tensor<131072x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x640xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
    return %2 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<131072x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<131072x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    return %4 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<128x160x256xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<32768x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x160x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x160x256xf32>) -> tensor<128x256x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x160xf32>) -> tensor<32768x160xf32>
    %6 = stablehlo.dot %5, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<32768x160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
    return %6 : tensor<32768x320xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<640x1024x32xf32>, %arg1: tensor<640x32x256xf32>) -> tensor<640x1024x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640x1024x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<640x1024x256xf32>
    return %2 : tensor<640x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<640x1024x256xf32>, %arg1: tensor<640x256x32xf32>) -> tensor<640x1024x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
    return %0 : tensor<640x1024x32xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    return %0 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<131072x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<131072x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<131072x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
    return %4 : tensor<131072x640xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<128x640x1024xf32>, %arg1: tensor<160x640xf32>) -> tensor<131072x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x640x1024xf32>) -> tensor<128x1024x640xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x640xf32>) -> tensor<131072x640xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x640xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
    return %2 : tensor<131072x160xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<1024x256x32xf32>, %arg1: tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x256x256xf32>
    return %2 : tensor<1024x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<1024x256x256xf32>, %arg1: tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
    return %0 : tensor<1024x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %0 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<32768x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
    return %4 : tensor<32768x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x1024x256xf32>) -> tensor<128x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x1024xf32>) -> tensor<32768x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x1024xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
    return %2 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<1024x256x32xf32>, %arg1: tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x256x256xf32>
    return %2 : tensor<1024x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %4 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<1024x256x256xf32>, %arg1: tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
    return %0 : tensor<1024x256x32xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    return %0 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<32768x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<32768x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<32768x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
    return %4 : tensor<32768x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<256x1024xf32>) -> tensor<32768x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<128x1024x256xf32>) -> tensor<128x256x1024xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x1024xf32>) -> tensor<32768x1024xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x1024xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
    return %2 : tensor<32768x256xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<128x256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x256x256xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<128x256x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x256xf32>, tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    return %6 : tensor<128x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<128x1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<256x160xf32>) -> tensor<128x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x1024x160xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<128x160x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1024x160xf32>, tensor<128x160x256xf32>) -> tensor<128x1024x256xf32>
    return %6 : tensor<128x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<128x4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<128x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x4096x64xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<128x64x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x64xf32>, tensor<128x64x256xf32>) -> tensor<128x4096x256xf32>
    return %6 : tensor<128x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<128x16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x32xf32>) -> tensor<128x16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x16384x32xf32>
    %4 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<128x32x256xf32>
    %6 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
    return %6 : tensor<128x16384x256xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.60(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<128x256x256xf32>, %arg3: tensor<32768x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<32768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %6 = stablehlo.add %5, %arg2 : tensor<128x256x256xf32>
    %7 = stablehlo.add %2, %6 : tensor<128x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<128x256x256xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<128x256x256xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %17 = stablehlo.add %15, %16 : tensor<128x256xf32>
    %18 = stablehlo.rsqrt %17 : tensor<128x256xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<128x256x256xf32>
    return %20 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.62(%arg0: tensor<128x256x256xf32>, %arg1: tensor<32768x256xf32>, %arg2: tensor<256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<128x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<128x256x256xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<128x256x256xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<128x256x256xf32>
    return %16 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.64(%arg0: tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x256x256xf32>, tensor<f32>) -> tensor<1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<128x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x8x256x256xf32>, tensor<f32>) -> tensor<128x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x8x256x256xf32>
    return %8 : tensor<128x8x256x256xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.66(%arg0: tensor<128x256x256xf32>) -> tensor<128x256x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<128x256x256xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<128x256x256xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<128x256x256xf32>
    return %12 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.68(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<128x256x16x16xf32>, %arg3: tensor<32768x256xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<128x256x256xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<128x256x256xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x256xf32>
    %17 = stablehlo.rsqrt %16 : tensor<128x256xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<128x256x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x256x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x256x256xf32>
    %24 = stablehlo.add %2, %23 : tensor<128x256x256xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x256xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<128x256x256xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<128x256x256xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<128x256xf32>
    %32 = stablehlo.add %31, %15 : tensor<128x256xf32>
    %33 = stablehlo.rsqrt %32 : tensor<128x256xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<128x256x256xf32>
    return %35 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.70(%arg0: tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x256x256xf32>, tensor<f32>) -> tensor<1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<128x8x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x8x256x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x8x256x256xf32>, tensor<f32>) -> tensor<128x8x256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x8x256x256xf32>
    return %8 : tensor<128x8x256x256xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.72(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<128x16x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<128x256x256xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<128x256x256xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x256xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<128x256x256xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x256x256xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %20 = stablehlo.add %18, %19 : tensor<128x256x256xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<128x256xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<128x256x256xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<128x256x256xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<128x256xf32>
    %28 = stablehlo.add %27, %12 : tensor<128x256xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128x256xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<128x256x256xf32>
    return %31 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.77(%arg0: tensor<128x1024x160xf32>, %arg1: tensor<131072x160xf32>, %arg2: tensor<160xf32>) -> tensor<128x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<128x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<128x1024x160xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<128x1024x160xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<128x1024x160xf32>
    return %16 : tensor<128x1024x160xf32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.79(%arg0: tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<640x1024x256xf32>, tensor<f32>) -> tensor<640x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<128x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x5x1024x256xf32>, tensor<f32>) -> tensor<128x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x5x1024x256xf32>
    return %8 : tensor<128x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.81(%arg0: tensor<128x1024x160xf32>) -> tensor<128x1024x160xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<128x1024x160xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<128x1024x160xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<128x1024x160xf32>
    return %12 : tensor<128x1024x160xf32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.83(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<128x160x32x32xf32>, %arg3: tensor<131072x160xf32>, %arg4: tensor<160xf32>) -> tensor<128x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<128x1024x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<128x1024x160xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x1024xf32>
    %17 = stablehlo.rsqrt %16 : tensor<128x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<128x1024x160xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x1024x160xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x1024x160xf32>
    %24 = stablehlo.add %2, %23 : tensor<128x1024x160xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x1024xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<128x1024x160xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<128x1024x160xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<128x1024xf32>
    %32 = stablehlo.add %31, %15 : tensor<128x1024xf32>
    %33 = stablehlo.rsqrt %32 : tensor<128x1024xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<128x1024x160xf32>
    return %35 : tensor<128x1024x160xf32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.85(%arg0: tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<640x1024x256xf32>, tensor<f32>) -> tensor<640x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<128x5x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x5x1024x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x5x1024x256xf32>, tensor<f32>) -> tensor<128x5x1024xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x5x1024x256xf32>
    return %8 : tensor<128x5x1024x256xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.87(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<128x160x32x32xf32>) -> tensor<128x1024x160xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<128x32x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<128x1024x160xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<128x1024x160xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x1024xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x1024xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<128x1024x160xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x1024x160xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %20 = stablehlo.add %18, %19 : tensor<128x1024x160xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<128x1024xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<128x1024x160xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<128x1024x160xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<128x1024xf32>
    %28 = stablehlo.add %27, %12 : tensor<128x1024xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128x1024xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<128x1024x160xf32>
    return %31 : tensor<128x1024x160xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.92(%arg0: tensor<128x4096x64xf32>, %arg1: tensor<524288x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<128x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<128x4096x64xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<128x4096x64xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<128x4096x64xf32>
    return %16 : tensor<128x4096x64xf32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.94(%arg0: tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x4096x256xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<128x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x2x4096x256xf32>, tensor<f32>) -> tensor<128x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x2x4096x256xf32>
    return %8 : tensor<128x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.96(%arg0: tensor<128x4096x64xf32>) -> tensor<128x4096x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<128x4096x64xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<128x4096x64xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<128x4096x64xf32>
    return %12 : tensor<128x4096x64xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.98(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<128x64x64x64xf32>, %arg3: tensor<524288x64xf32>, %arg4: tensor<64xf32>) -> tensor<128x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<128x4096x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<128x4096x64xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x4096xf32>
    %17 = stablehlo.rsqrt %16 : tensor<128x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<128x4096x64xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x4096x64xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x4096x64xf32>
    %24 = stablehlo.add %2, %23 : tensor<128x4096x64xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x4096xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<128x4096x64xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<128x4096x64xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<128x4096xf32>
    %32 = stablehlo.add %31, %15 : tensor<128x4096xf32>
    %33 = stablehlo.rsqrt %32 : tensor<128x4096xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<128x4096x64xf32>
    return %35 : tensor<128x4096x64xf32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.100(%arg0: tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x4096x256xf32>, tensor<f32>) -> tensor<256x4096xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<128x2x4096xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x2x4096x256xf32>
    %5 = stablehlo.exponential %4 : tensor<128x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x2x4096x256xf32>, tensor<f32>) -> tensor<128x2x4096xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
    %8 = stablehlo.divide %5, %7 : tensor<128x2x4096x256xf32>
    return %8 : tensor<128x2x4096x256xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.102(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<128x64x64x64xf32>) -> tensor<128x4096x64xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<128x64x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<128x4096x64xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<128x4096x64xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x4096xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x4096xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<128x4096x64xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x4096x64xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %20 = stablehlo.add %18, %19 : tensor<128x4096x64xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<128x4096xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<128x4096x64xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<128x4096x64xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<128x4096xf32>
    %28 = stablehlo.add %27, %12 : tensor<128x4096xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128x4096xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<128x4096x64xf32>
    return %31 : tensor<128x4096x64xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.107(%arg0: tensor<128x16384x32xf32>, %arg1: tensor<2097152x32xf32>, %arg2: tensor<32xf32>) -> tensor<128x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<2097152x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<128x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128x16384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<128x16384x32xf32>
    %9 = stablehlo.multiply %8, %8 : tensor<128x16384x32xf32>
    %10 = stablehlo.reduce(%9 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %11 = stablehlo.multiply %10, %5 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %16 = stablehlo.multiply %8, %15 : tensor<128x16384x32xf32>
    return %16 : tensor<128x16384x32xf32>
  }
  func.func private @region_0.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.109(%arg0: tensor<128x16384x256xf32>) -> tensor<128x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<128x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<128x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<128x16384x256xf32>
    return %6 : tensor<128x16384x256xf32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.111(%arg0: tensor<128x16384x32xf32>) -> tensor<128x16384x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<128x16384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<128x16384x32xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<128x16384x32xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %12 = stablehlo.multiply %4, %11 : tensor<128x16384x32xf32>
    return %12 : tensor<128x16384x32xf32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.113(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<128x32x128x128xf32>, %arg3: tensor<2097152x32xf32>, %arg4: tensor<32xf32>) -> tensor<128x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.add %0, %arg3 : tensor<2097152x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %3 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<128x16384x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<128x16384x32xf32>
    %12 = stablehlo.reduce(%11 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x16384xf32>
    %17 = stablehlo.rsqrt %16 : tensor<128x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %19 = stablehlo.multiply %10, %18 : tensor<128x16384x32xf32>
    %20 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128x16384x32xf32>
    %22 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %23 = stablehlo.add %21, %22 : tensor<128x16384x32xf32>
    %24 = stablehlo.add %2, %23 : tensor<128x16384x32xf32>
    %25 = stablehlo.reduce(%24 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x16384xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %28 = stablehlo.subtract %24, %27 : tensor<128x16384x32xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<128x16384x32xf32>
    %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %31 = stablehlo.multiply %30, %13 : tensor<128x16384xf32>
    %32 = stablehlo.add %31, %15 : tensor<128x16384xf32>
    %33 = stablehlo.rsqrt %32 : tensor<128x16384xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %35 = stablehlo.multiply %28, %34 : tensor<128x16384x32xf32>
    return %35 : tensor<128x16384x32xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.115(%arg0: tensor<128x16384x256xf32>) -> tensor<128x16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<128x16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<128x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<128x16384x256xf32>
    return %6 : tensor<128x16384x256xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.117(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<128x32x128x128xf32>) -> tensor<128x16384x32xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%arg2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<128x128x128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<128x16384x32xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<128x16384x32xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %13 = stablehlo.add %11, %12 : tensor<128x16384xf32>
    %14 = stablehlo.rsqrt %13 : tensor<128x16384xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %16 = stablehlo.multiply %7, %15 : tensor<128x16384x32xf32>
    %17 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x16384x32xf32>
    %19 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %20 = stablehlo.add %18, %19 : tensor<128x16384x32xf32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %22 = stablehlo.multiply %21, %10 : tensor<128x16384xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %24 = stablehlo.subtract %20, %23 : tensor<128x16384x32xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<128x16384x32xf32>
    %26 = stablehlo.reduce(%25 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %27 = stablehlo.multiply %26, %10 : tensor<128x16384xf32>
    %28 = stablehlo.add %27, %12 : tensor<128x16384xf32>
    %29 = stablehlo.rsqrt %28 : tensor<128x16384xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %31 = stablehlo.multiply %24, %30 : tensor<128x16384x32xf32>
    return %31 : tensor<128x16384x32xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<1024x256x32xf32>) -> tensor<128x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x32xf32>) -> tensor<128x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x8x256x32xf32>) -> tensor<128x256x8x32xf32>
    return %1 : tensor<128x256x8x32xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<1024x256x32xf32>) -> tensor<128x256x8x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x32xf32>) -> tensor<128x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x8x256x32xf32>) -> tensor<128x256x8x32xf32>
    return %1 : tensor<128x256x8x32xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<640x1024x32xf32>) -> tensor<128x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x32xf32>) -> tensor<128x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x5x1024x32xf32>) -> tensor<128x1024x5x32xf32>
    return %1 : tensor<128x1024x5x32xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<640x1024x32xf32>) -> tensor<128x1024x5x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x32xf32>) -> tensor<128x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x5x1024x32xf32>) -> tensor<128x1024x5x32xf32>
    return %1 : tensor<128x1024x5x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<256x4096x32xf32>) -> tensor<128x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x32xf32>) -> tensor<128x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x2x4096x32xf32>) -> tensor<128x4096x2x32xf32>
    return %1 : tensor<128x4096x2x32xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<256x4096x32xf32>) -> tensor<128x4096x2x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x32xf32>) -> tensor<128x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] : (tensor<128x2x4096x32xf32>) -> tensor<128x4096x2x32xf32>
    return %1 : tensor<128x4096x2x32xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<128x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<128x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<128x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<128x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<128x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x1024x256xf32>
    return %98 : tensor<128x1024x256xf32>
  }
  func.func private @fused_transpose.6(%arg0: tensor<32768x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<128x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<32768x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x1024xf32>) -> tensor<128x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x256x1024xf32>) -> tensor<128x1024x256xf32>
    return %3 : tensor<128x1024x256xf32>
  }
  func.func private @fused_transpose.7(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
    return %3 : tensor<128x8x256x32xf32>
  }
  func.func private @fused_transpose.8(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<128x256x8x32xf32>) -> tensor<128x8x32x256xf32>
    return %3 : tensor<128x8x32x256xf32>
  }
  func.func private @fused_transpose.9(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
    return %3 : tensor<128x8x256x32xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<128x256x16x16xf32>, %arg5: tensor<32768x256xf32>, %arg6: tensor<256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<32768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x16x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<128x256x256xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<128x256x256xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x256xf32>
    %20 = stablehlo.rsqrt %19 : tensor<128x256xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<128x256x256xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x256x256xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x256x256xf32>
    %27 = stablehlo.add %5, %26 : tensor<128x256x256xf32>
    %28 = stablehlo.add %2, %27 : tensor<128x256x256xf32>
    return %28 : tensor<128x256x256xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<128x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<128x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<128x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<128x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<128x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<128x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x1024x256xf32>
    return %98 : tensor<128x1024x256xf32>
  }
  func.func private @fused_transpose.10(%arg0: tensor<32768x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<128x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<32768x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x1024xf32>) -> tensor<128x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x256x1024xf32>) -> tensor<128x1024x256xf32>
    return %3 : tensor<128x1024x256xf32>
  }
  func.func private @fused_transpose.11(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
    return %3 : tensor<128x8x256x32xf32>
  }
  func.func private @fused_transpose.12(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] : (tensor<128x256x8x32xf32>) -> tensor<128x8x32x256xf32>
    return %3 : tensor<128x8x32x256xf32>
  }
  func.func private @fused_transpose.13(%arg0: tensor<32768x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x8x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<32768x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
    return %3 : tensor<128x8x256x32xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<128x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<128x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<128x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<128x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<128x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<128x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<128x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x640x1024xf32>
    return %98 : tensor<128x640x1024xf32>
  }
  func.func private @fused_transpose.14(%arg0: tensor<131072x640xf32>, %arg1: tensor<640xf32>) -> tensor<128x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<131072x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<131072x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x640xf32>) -> tensor<128x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x1024x640xf32>) -> tensor<128x640x1024xf32>
    return %3 : tensor<128x640x1024xf32>
  }
  func.func private @fused_transpose.15(%arg0: tensor<32768x320xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 160:320] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<128x256x5x32xf32>) -> tensor<128x5x256x32xf32>
    return %4 : tensor<128x5x256x32xf32>
  }
  func.func private @fused_transpose.16(%arg0: tensor<32768x320xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 0:160] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<128x256x5x32xf32>) -> tensor<128x5x32x256xf32>
    return %4 : tensor<128x5x32x256xf32>
  }
  func.func private @fused_transpose.17(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x1024x5x32xf32>) -> tensor<128x5x1024x32xf32>
    return %3 : tensor<128x5x1024x32xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<128x160x32x32xf32>, %arg5: tensor<131072x160xf32>, %arg6: tensor<160xf32>) -> tensor<128x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<131072x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x32x32xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<128x1024x160xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<128x1024x160xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x1024xf32>
    %20 = stablehlo.rsqrt %19 : tensor<128x1024xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<128x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x1024x160xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x1024x160xf32>
    %27 = stablehlo.add %5, %26 : tensor<128x1024x160xf32>
    %28 = stablehlo.add %2, %27 : tensor<128x1024x160xf32>
    return %28 : tensor<128x1024x160xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<128x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<128x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<128x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<128x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<128x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<128x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<128x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x640x1024xf32>
    return %98 : tensor<128x640x1024xf32>
  }
  func.func private @fused_transpose.18(%arg0: tensor<131072x640xf32>, %arg1: tensor<640xf32>) -> tensor<128x640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<131072x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<131072x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x640xf32>) -> tensor<128x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x1024x640xf32>) -> tensor<128x640x1024xf32>
    return %3 : tensor<128x640x1024xf32>
  }
  func.func private @fused_transpose.19(%arg0: tensor<32768x320xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 160:320] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<128x256x5x32xf32>) -> tensor<128x5x256x32xf32>
    return %4 : tensor<128x5x256x32xf32>
  }
  func.func private @fused_transpose.20(%arg0: tensor<32768x320xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 0:160] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x160xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<128x256x5x32xf32>) -> tensor<128x5x32x256xf32>
    return %4 : tensor<128x5x32x256xf32>
  }
  func.func private @fused_transpose.21(%arg0: tensor<131072x160xf32>, %arg1: tensor<160xf32>) -> tensor<128x5x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x1024x5x32xf32>) -> tensor<128x5x1024x32xf32>
    return %3 : tensor<128x5x1024x32xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<128x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<128x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<128x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<128x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<128x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<128x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x256x4096xf32>
    return %98 : tensor<128x256x4096xf32>
  }
  func.func private @fused_transpose.22(%arg0: tensor<524288x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<524288x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<524288x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<128x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x4096x256xf32>) -> tensor<128x256x4096xf32>
    return %3 : tensor<128x256x4096xf32>
  }
  func.func private @fused_transpose.23(%arg0: tensor<32768x128xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 64:128] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<128x256x2x32xf32>) -> tensor<128x2x256x32xf32>
    return %4 : tensor<128x2x256x32xf32>
  }
  func.func private @fused_transpose.24(%arg0: tensor<32768x128xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 0:64] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<128x256x2x32xf32>) -> tensor<128x2x32x256xf32>
    return %4 : tensor<128x2x32x256xf32>
  }
  func.func private @fused_transpose.25(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x4096x2x32xf32>) -> tensor<128x2x4096x32xf32>
    return %3 : tensor<128x2x4096x32xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<128x64x64x64xf32>, %arg5: tensor<524288x64xf32>, %arg6: tensor<64xf32>) -> tensor<128x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<524288x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x64x64xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<128x4096x64xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<128x4096x64xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x4096xf32>
    %20 = stablehlo.rsqrt %19 : tensor<128x4096xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<128x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x4096x64xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x4096x64xf32>
    %27 = stablehlo.add %5, %26 : tensor<128x4096x64xf32>
    %28 = stablehlo.add %2, %27 : tensor<128x4096x64xf32>
    return %28 : tensor<128x4096x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<128x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<128x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<128x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<128x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<128x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<128x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x256x4096xf32>
    return %98 : tensor<128x256x4096xf32>
  }
  func.func private @fused_transpose.26(%arg0: tensor<524288x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<524288x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<524288x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<128x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x4096x256xf32>) -> tensor<128x256x4096xf32>
    return %3 : tensor<128x256x4096xf32>
  }
  func.func private @fused_transpose.27(%arg0: tensor<32768x128xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 64:128] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<128x256x2x32xf32>) -> tensor<128x2x256x32xf32>
    return %4 : tensor<128x2x256x32xf32>
  }
  func.func private @fused_transpose.28(%arg0: tensor<32768x128xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
    %1 = stablehlo.slice %arg0 [0:32768, 0:64] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
    %2 = stablehlo.add %0, %1 : tensor<32768x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 3, 1] : (tensor<128x256x2x32xf32>) -> tensor<128x2x32x256xf32>
    return %4 : tensor<128x2x32x256xf32>
  }
  func.func private @fused_transpose.29(%arg0: tensor<524288x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x2x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<128x4096x2x32xf32>) -> tensor<128x2x4096x32xf32>
    return %3 : tensor<128x2x4096x32xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<128x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<128x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<128x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<128x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<128x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x128x16384xf32>
    return %98 : tensor<128x128x16384xf32>
  }
  func.func private @fused_transpose.30(%arg0: tensor<2097152x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<2097152x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<2097152x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<128x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x16384x128xf32>) -> tensor<128x128x16384xf32>
    return %3 : tensor<128x128x16384xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<2097152x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<128x32x128x128xf32>, %arg5: tensor<2097152x32xf32>, %arg6: tensor<32xf32>) -> tensor<128x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<2097152x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<2097152x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%arg4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x128x128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %13 = stablehlo.subtract %7, %12 : tensor<128x16384x32xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<128x16384x32xf32>
    %15 = stablehlo.reduce(%14 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x16384xf32>
    %20 = stablehlo.rsqrt %19 : tensor<128x16384xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %22 = stablehlo.multiply %13, %21 : tensor<128x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x16384x32xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x16384x32xf32>
    %27 = stablehlo.add %5, %26 : tensor<128x16384x32xf32>
    %28 = stablehlo.add %2, %27 : tensor<128x16384x32xf32>
    return %28 : tensor<128x16384x32xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<128x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<128x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<128x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<128x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<128x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<128x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<128x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<128x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<128x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<128x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<128x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<128x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<128x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<128x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<128x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<128x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<128x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<128x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<128x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<128x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<128x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<128x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<128x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<128x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<128x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<128x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<128x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<128x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<128x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<128x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<128x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<128x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<128x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<128x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<128x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<128x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<128x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<128x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<128x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<128x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<128x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<128x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<128x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<128x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<128x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<128x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<128x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<128x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<128x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<128x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<128x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<128x128x16384xf32>
    return %98 : tensor<128x128x16384xf32>
  }
  func.func private @fused_transpose.31(%arg0: tensor<2097152x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<2097152x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<2097152x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<128x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<128x16384x128xf32>) -> tensor<128x128x16384xf32>
    return %3 : tensor<128x128x16384xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<128x256x128x128xf32>, %arg4: tensor<256xf32>) -> tensor<128x256x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<128x256x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<128x256x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x256x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128x256x128x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x128x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<128x256x128x128xf32>
    return %9 : tensor<128x256x128x128xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128x1xf32>, %arg3: tensor<128x1xf32>, %arg4: tensor<128x256x256xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x1024x256xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x16384x256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<128x4096x256xf32>, %arg13: tensor<256xf32>) -> tensor<128x1024x128x128xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,16,16]{1,3,2,0}"} : (tensor<128x256x256xf32>) -> tensor<128x256x16x16xf32>
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
    %c = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %14 = stablehlo.compare LT, %12, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_3 = stablehlo.constant dense<16> : tensor<i32>
    %15 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %16 = stablehlo.add %12, %15 : tensor<128x1xi32>
    %17 = stablehlo.select %14, %16, %12 : tensor<128x1xi1>, tensor<128x1xi32>
    %18 = mhlo.bitcast %17 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %20 = stablehlo.convert %10 : (tensor<128xf32>) -> tensor<128xi32>
    %21 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %22 = stablehlo.compare LT, %20, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %23 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %24 = stablehlo.add %20, %23 : tensor<128xi32>
    %25 = stablehlo.select %22, %24, %20 : tensor<128xi1>, tensor<128xi32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %27 = stablehlo.concatenate %19, %26, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %29 = "stablehlo.gather"(%0, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %c_4 = stablehlo.constant dense<15> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %c_5 = stablehlo.constant dense<1> : tensor<i32>
    %32 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %33 = stablehlo.add %20, %32 : tensor<128xi32>
    %34 = stablehlo.minimum %31, %33 : tensor<128xi32>
    %35 = stablehlo.compare LT, %34, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %36 = stablehlo.add %34, %23 : tensor<128xi32>
    %37 = stablehlo.select %35, %36, %34 : tensor<128xi1>, tensor<128xi32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %39 = stablehlo.concatenate %19, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %40 = mhlo.bitcast %39 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %41 = "stablehlo.gather"(%0, %40) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %43 = stablehlo.subtract %42, %30 : tensor<128x256x128x128xf32>
    %44 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<128x256x128x128xf32>
    %46 = stablehlo.add %30, %45 : tensor<128x256x128x128xf32>
    %47 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %48 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %49 = stablehlo.add %12, %48 : tensor<128x1xi32>
    %50 = stablehlo.minimum %47, %49 : tensor<128x1xi32>
    %51 = stablehlo.compare LT, %50, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %52 = stablehlo.add %50, %15 : tensor<128x1xi32>
    %53 = stablehlo.select %51, %52, %50 : tensor<128x1xi1>, tensor<128x1xi32>
    %54 = mhlo.bitcast %53 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %56 = stablehlo.concatenate %55, %26, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %57 = mhlo.bitcast %56 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %58 = "stablehlo.gather"(%0, %57) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %60 = stablehlo.concatenate %55, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %62 = "stablehlo.gather"(%0, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %64 = stablehlo.subtract %63, %59 : tensor<128x256x128x128xf32>
    %65 = stablehlo.multiply %64, %44 : tensor<128x256x128x128xf32>
    %66 = stablehlo.add %59, %65 : tensor<128x256x128x128xf32>
    %67 = stablehlo.subtract %66, %46 : tensor<128x256x128x128xf32>
    %68 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %70 = stablehlo.multiply %67, %69 : tensor<128x256x128x128xf32>
    %71 = stablehlo.add %46, %70 : tensor<128x256x128x128xf32>
    %72 = mhlo.bitcast %arg8 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,32,32]{1,3,2,0}"} : (tensor<128x1024x256xf32>) -> tensor<128x256x32x32xf32>
    %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %74 = stablehlo.multiply %5, %73 : tensor<128xf32>
    %75 = stablehlo.add %74, %8 : tensor<128xf32>
    %76 = stablehlo.maximum %1, %75 : tensor<128xf32>
    %77 = mhlo.bitcast %76 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %78 = stablehlo.convert %77 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %79 = stablehlo.compare LT, %78, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_7 = stablehlo.constant dense<32> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %81 = stablehlo.add %78, %80 : tensor<128x1xi32>
    %82 = stablehlo.select %79, %81, %78 : tensor<128x1xi1>, tensor<128x1xi32>
    %83 = mhlo.bitcast %82 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %85 = stablehlo.convert %76 : (tensor<128xf32>) -> tensor<128xi32>
    %86 = stablehlo.compare LT, %85, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %87 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %88 = stablehlo.add %85, %87 : tensor<128xi32>
    %89 = stablehlo.select %86, %88, %85 : tensor<128xi1>, tensor<128xi32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %91 = stablehlo.concatenate %84, %90, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %92 = mhlo.bitcast %91 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %93 = "stablehlo.gather"(%72, %92) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %c_8 = stablehlo.constant dense<31> : tensor<i32>
    %95 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %96 = stablehlo.add %85, %32 : tensor<128xi32>
    %97 = stablehlo.minimum %95, %96 : tensor<128xi32>
    %98 = stablehlo.compare LT, %97, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %99 = stablehlo.add %97, %87 : tensor<128xi32>
    %100 = stablehlo.select %98, %99, %97 : tensor<128xi1>, tensor<128xi32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %102 = stablehlo.concatenate %84, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %104 = "stablehlo.gather"(%72, %103) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %106 = stablehlo.subtract %105, %94 : tensor<128x256x128x128xf32>
    %107 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<128x256x128x128xf32>
    %109 = stablehlo.add %94, %108 : tensor<128x256x128x128xf32>
    %110 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %111 = stablehlo.add %78, %48 : tensor<128x1xi32>
    %112 = stablehlo.minimum %110, %111 : tensor<128x1xi32>
    %113 = stablehlo.compare LT, %112, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %114 = stablehlo.add %112, %80 : tensor<128x1xi32>
    %115 = stablehlo.select %113, %114, %112 : tensor<128x1xi1>, tensor<128x1xi32>
    %116 = mhlo.bitcast %115 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %118 = stablehlo.concatenate %117, %90, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %120 = "stablehlo.gather"(%72, %119) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %122 = stablehlo.concatenate %117, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %124 = "stablehlo.gather"(%72, %123) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %126 = stablehlo.subtract %125, %121 : tensor<128x256x128x128xf32>
    %127 = stablehlo.multiply %126, %107 : tensor<128x256x128x128xf32>
    %128 = stablehlo.add %121, %127 : tensor<128x256x128x128xf32>
    %129 = stablehlo.subtract %128, %109 : tensor<128x256x128x128xf32>
    %130 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %132 = stablehlo.multiply %129, %131 : tensor<128x256x128x128xf32>
    %133 = stablehlo.add %109, %132 : tensor<128x256x128x128xf32>
    %134 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<256xf32>) -> tensor<128x4096x256xf32>
    %135 = stablehlo.add %arg12, %134 : tensor<128x4096x256xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,64,64]{1,3,2,0}"} : (tensor<128x4096x256xf32>) -> tensor<128x256x64x64xf32>
    %137 = stablehlo.multiply %5, %4 : tensor<128xf32>
    %138 = stablehlo.add %137, %8 : tensor<128xf32>
    %139 = stablehlo.maximum %1, %138 : tensor<128xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %141 = stablehlo.convert %140 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %142 = stablehlo.compare LT, %141, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_9 = stablehlo.constant dense<64> : tensor<i32>
    %143 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %144 = stablehlo.add %141, %143 : tensor<128x1xi32>
    %145 = stablehlo.select %142, %144, %141 : tensor<128x1xi1>, tensor<128x1xi32>
    %146 = mhlo.bitcast %145 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %148 = stablehlo.convert %139 : (tensor<128xf32>) -> tensor<128xi32>
    %149 = stablehlo.compare LT, %148, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %150 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %151 = stablehlo.add %148, %150 : tensor<128xi32>
    %152 = stablehlo.select %149, %151, %148 : tensor<128xi1>, tensor<128xi32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %154 = stablehlo.concatenate %147, %153, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %156 = "stablehlo.gather"(%136, %155) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %c_10 = stablehlo.constant dense<63> : tensor<i32>
    %158 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %159 = stablehlo.add %148, %32 : tensor<128xi32>
    %160 = stablehlo.minimum %158, %159 : tensor<128xi32>
    %161 = stablehlo.compare LT, %160, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %162 = stablehlo.add %160, %150 : tensor<128xi32>
    %163 = stablehlo.select %161, %162, %160 : tensor<128xi1>, tensor<128xi32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %165 = stablehlo.concatenate %147, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %167 = "stablehlo.gather"(%136, %166) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %169 = stablehlo.subtract %168, %157 : tensor<128x256x128x128xf32>
    %170 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<128x256x128x128xf32>
    %172 = stablehlo.add %157, %171 : tensor<128x256x128x128xf32>
    %173 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %174 = stablehlo.add %141, %48 : tensor<128x1xi32>
    %175 = stablehlo.minimum %173, %174 : tensor<128x1xi32>
    %176 = stablehlo.compare LT, %175, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %177 = stablehlo.add %175, %143 : tensor<128x1xi32>
    %178 = stablehlo.select %176, %177, %175 : tensor<128x1xi1>, tensor<128x1xi32>
    %179 = mhlo.bitcast %178 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %181 = stablehlo.concatenate %180, %153, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %182 = mhlo.bitcast %181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %183 = "stablehlo.gather"(%136, %182) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %185 = stablehlo.concatenate %180, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %187 = "stablehlo.gather"(%136, %186) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %189 = stablehlo.subtract %188, %184 : tensor<128x256x128x128xf32>
    %190 = stablehlo.multiply %189, %170 : tensor<128x256x128x128xf32>
    %191 = stablehlo.add %184, %190 : tensor<128x256x128x128xf32>
    %192 = stablehlo.subtract %191, %172 : tensor<128x256x128x128xf32>
    %193 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %195 = stablehlo.multiply %192, %194 : tensor<128x256x128x128xf32>
    %196 = stablehlo.add %172, %195 : tensor<128x256x128x128xf32>
    %197 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<256xf32>) -> tensor<128x16384x256xf32>
    %198 = stablehlo.add %arg10, %197 : tensor<128x16384x256xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,128,128]{1,3,2,0}"} : (tensor<128x16384x256xf32>) -> tensor<128x256x128x128xf32>
    %200 = stablehlo.maximum %1, %3 : tensor<128xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %202 = stablehlo.convert %201 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %203 = stablehlo.compare LT, %202, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_11 = stablehlo.constant dense<128> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %205 = stablehlo.add %202, %204 : tensor<128x1xi32>
    %206 = stablehlo.select %203, %205, %202 : tensor<128x1xi1>, tensor<128x1xi32>
    %207 = mhlo.bitcast %206 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %209 = stablehlo.convert %200 : (tensor<128xf32>) -> tensor<128xi32>
    %210 = stablehlo.compare LT, %209, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %211 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %212 = stablehlo.add %209, %211 : tensor<128xi32>
    %213 = stablehlo.select %210, %212, %209 : tensor<128xi1>, tensor<128xi32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %215 = stablehlo.concatenate %208, %214, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %217 = "stablehlo.gather"(%199, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %c_12 = stablehlo.constant dense<127> : tensor<i32>
    %219 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %220 = stablehlo.add %209, %32 : tensor<128xi32>
    %221 = stablehlo.minimum %219, %220 : tensor<128xi32>
    %222 = stablehlo.compare LT, %221, %21 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %223 = stablehlo.add %221, %211 : tensor<128xi32>
    %224 = stablehlo.select %222, %223, %221 : tensor<128xi1>, tensor<128xi32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %226 = stablehlo.concatenate %208, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %228 = "stablehlo.gather"(%199, %227) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %230 = stablehlo.subtract %229, %218 : tensor<128x256x128x128xf32>
    %231 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %232 = stablehlo.multiply %230, %231 : tensor<128x256x128x128xf32>
    %233 = stablehlo.add %218, %232 : tensor<128x256x128x128xf32>
    %234 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %235 = stablehlo.add %202, %48 : tensor<128x1xi32>
    %236 = stablehlo.minimum %234, %235 : tensor<128x1xi32>
    %237 = stablehlo.compare LT, %236, %13 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %238 = stablehlo.add %236, %204 : tensor<128x1xi32>
    %239 = stablehlo.select %237, %238, %236 : tensor<128x1xi1>, tensor<128x1xi32>
    %240 = mhlo.bitcast %239 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %242 = stablehlo.concatenate %241, %214, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %244 = "stablehlo.gather"(%199, %243) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %246 = stablehlo.concatenate %241, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %248 = "stablehlo.gather"(%199, %247) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
    %250 = stablehlo.subtract %249, %245 : tensor<128x256x128x128xf32>
    %251 = stablehlo.multiply %250, %231 : tensor<128x256x128x128xf32>
    %252 = stablehlo.add %245, %251 : tensor<128x256x128x128xf32>
    %253 = stablehlo.subtract %252, %233 : tensor<128x256x128x128xf32>
    %254 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
    %256 = stablehlo.multiply %253, %255 : tensor<128x256x128x128xf32>
    %257 = stablehlo.add %233, %256 : tensor<128x256x128x128xf32>
    %258 = stablehlo.concatenate %71, %133, %196, %257, dim = 1 : (tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>) -> tensor<128x1024x128x128xf32>
    return %258 : tensor<128x1024x128x128xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.563(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<2097152x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<128x16384x32xf32>, %arg5: tensor<2097152x32xf32>, %arg6: tensor<32xf32>) -> (tensor<128x32x16384xf32>, tensor<128x16384x32xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<2097152x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<2097152x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<128x16384x32xf32>
    %7 = stablehlo.add %2, %6 : tensor<128x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x16384xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<128x16384x32xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<128x16384x32xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<128x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %17 = stablehlo.add %15, %16 : tensor<128x16384xf32>
    %18 = stablehlo.rsqrt %17 : tensor<128x16384xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<128x16384x32xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x16384x32xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %24 = stablehlo.add %22, %23 : tensor<128x16384x32xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
    return %25, %20 : tensor<128x32x16384xf32>, tensor<128x16384x32xf32>
  }
  func.func private @fused_transpose.32(%arg0: tensor<32xf32>, %arg1: tensor<128x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<128x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
    return %4 : tensor<128x32x16384xf32>
  }
  func.func private @fused_transpose.33(%arg0: tensor<32xf32>, %arg1: tensor<128x16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<128x32x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x16384x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
    return %4 : tensor<128x32x16384xf32>
  }
  func.func private @fused_add(%arg0: tensor<128x1024x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<128x1024x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128x1024x256xf32>
    return %1 : tensor<128x1024x256xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.682(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<524288x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<128x4096x64xf32>, %arg5: tensor<524288x64xf32>, %arg6: tensor<64xf32>) -> (tensor<128x64x4096xf32>, tensor<128x4096x64xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<524288x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<524288x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<128x4096x64xf32>
    %7 = stablehlo.add %2, %6 : tensor<128x4096x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x4096xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<128x4096x64xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<128x4096x64xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<128x4096xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
    %17 = stablehlo.add %15, %16 : tensor<128x4096xf32>
    %18 = stablehlo.rsqrt %17 : tensor<128x4096xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<128x4096x64xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x4096x64xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %24 = stablehlo.add %22, %23 : tensor<128x4096x64xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
    return %25, %20 : tensor<128x64x4096xf32>, tensor<128x4096x64xf32>
  }
  func.func private @fused_transpose.34(%arg0: tensor<64xf32>, %arg1: tensor<128x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
    return %4 : tensor<128x64x4096xf32>
  }
  func.func private @fused_transpose.35(%arg0: tensor<64xf32>, %arg1: tensor<128x4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x4096x64xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
    return %4 : tensor<128x64x4096xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.701(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<131072x160xf32>, %arg3: tensor<160xf32>, %arg4: tensor<128x1024x160xf32>, %arg5: tensor<131072x160xf32>, %arg6: tensor<160xf32>) -> (tensor<128x160x1024xf32>, tensor<128x1024x160xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<131072x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
    %4 = stablehlo.add %3, %arg5 : tensor<131072x160xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
    %6 = stablehlo.add %5, %arg4 : tensor<128x1024x160xf32>
    %7 = stablehlo.add %2, %6 : tensor<128x1024x160xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<128x1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %12 = stablehlo.subtract %7, %11 : tensor<128x1024x160xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<128x1024x160xf32>
    %14 = stablehlo.reduce(%13 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
    %15 = stablehlo.multiply %14, %9 : tensor<128x1024xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %17 = stablehlo.add %15, %16 : tensor<128x1024xf32>
    %18 = stablehlo.rsqrt %17 : tensor<128x1024xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
    %20 = stablehlo.multiply %12, %19 : tensor<128x1024x160xf32>
    %21 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x1024x160xf32>
    %23 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %24 = stablehlo.add %22, %23 : tensor<128x1024x160xf32>
    %25 = stablehlo.transpose %24, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
    return %25, %20 : tensor<128x160x1024xf32>, tensor<128x1024x160xf32>
  }
  func.func private @fused_transpose.36(%arg0: tensor<160xf32>, %arg1: tensor<128x1024x160xf32>, %arg2: tensor<160xf32>) -> tensor<128x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
    return %4 : tensor<128x160x1024xf32>
  }
  func.func private @fused_transpose.37(%arg0: tensor<160xf32>, %arg1: tensor<128x1024x160xf32>, %arg2: tensor<160xf32>) -> tensor<128x160x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x1024x160xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
    return %4 : tensor<128x160x1024xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<128x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<128x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128x256x256xf32>
    return %1 : tensor<128x256x256xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<128x16x16xf32>, %arg1: tensor<128x160x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %5 : tensor<128x256xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<128x16x16xf32>, %arg1: tensor<128x160x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x160x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %5 : tensor<128x256xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<128x256xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x160x256xf32>
    return %5 : tensor<128x160x256xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<128x256xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<128x160x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x160x256xf32>
    return %5 : tensor<128x160x256xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<128x64x16x16xf32>, %arg1: tensor<128x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x64x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<128x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %7 : tensor<128x256xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<128x64x16x16xf32>, %arg1: tensor<128x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x64x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<128x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %7 : tensor<128x256xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<128x64x16x16xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<128x64x256xf32>
    return %12 : tensor<128x64x256xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<128x64x16x16xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<128x64x256xf32>
    return %12 : tensor<128x64x256xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<128x32x16x16xf32>, %arg1: tensor<128x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<128x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x32x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %7 : tensor<128x256xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<128x32x16x16xf32>, %arg1: tensor<128x16x16xf32>) -> tensor<128x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x32x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<128x32x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x32x256xf32>, tensor<f32>) -> tensor<128x256xf32>
    return %7 : tensor<128x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<128x32x16x16xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<128x32x256xf32>
    return %12 : tensor<128x32x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<128x32x16x16xf32>, %arg1: tensor<128x16x16xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x32x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %2 = stablehlo.multiply %arg1, %1 : tensor<128x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<128x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %7 = stablehlo.multiply %arg2, %6 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x256xf32>
    %10 = stablehlo.rsqrt %9 : tensor<128x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %12 = stablehlo.multiply %5, %11 : tensor<128x32x256xf32>
    return %12 : tensor<128x32x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<128x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x16x16xf32>
    return %1 : tensor<128x16x16xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<128x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x16x16xf32>
    return %1 : tensor<128x16x16xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x256xf32>
    return %4 : tensor<128x256xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x256xf32>
    return %4 : tensor<128x256xf32>
  }
  func.func private @fused_clamp() -> tensor<128xf32> {
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
    %10 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %11 = stablehlo.convert %10 : (tensor<128xi32>) -> tensor<128xf32>
    %12 = stablehlo.subtract %9, %11 : tensor<128xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %14 = stablehlo.clamp %0, %12, %13 : tensor<128xf32>
    return %14 : tensor<128xf32>
  }
  func.func private @fused_clamp.1() -> tensor<128xf32> {
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
    %10 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %11 = stablehlo.convert %10 : (tensor<128xi32>) -> tensor<128xf32>
    %12 = stablehlo.subtract %9, %11 : tensor<128xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %14 = stablehlo.clamp %0, %12, %13 : tensor<128xf32>
    return %14 : tensor<128xf32>
  }
  func.func private @fused_clamp.2() -> tensor<128xf32> {
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
    %9 = stablehlo.convert %8 : (tensor<128xf32>) -> tensor<128xi32>
    %10 = stablehlo.convert %9 : (tensor<128xi32>) -> tensor<128xf32>
    %11 = stablehlo.subtract %8, %10 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %13 = stablehlo.clamp %0, %11, %12 : tensor<128xf32>
    return %13 : tensor<128xf32>
  }
  func.func private @fused_clamp.3() -> tensor<128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %3 = stablehlo.maximum %0, %2 : tensor<128xf32>
    %4 = stablehlo.convert %3 : (tensor<128xf32>) -> tensor<128xi32>
    %5 = stablehlo.convert %4 : (tensor<128xi32>) -> tensor<128xf32>
    %6 = stablehlo.subtract %3, %5 : tensor<128xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.clamp %0, %6, %7 : tensor<128xf32>
    return %8 : tensor<128xf32>
  }
  func.func private @fused_clamp.4() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
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
    %13 = stablehlo.convert %12 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<128x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %16 = stablehlo.clamp %0, %14, %15 : tensor<128x1xf32>
    return %16 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.5() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
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
    %13 = stablehlo.convert %12 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<128x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %16 = stablehlo.clamp %0, %14, %15 : tensor<128x1xf32>
    return %16 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.6() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
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
    %12 = stablehlo.convert %11 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %13 = stablehlo.subtract %10, %12 : tensor<128x1xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %15 = stablehlo.clamp %0, %13, %14 : tensor<128x1xf32>
    return %15 : tensor<128x1xf32>
  }
  func.func private @fused_clamp.7() -> tensor<128x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %4 = stablehlo.maximum %1, %3 : tensor<128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %6 = stablehlo.convert %5 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %7 = stablehlo.convert %6 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %8 = stablehlo.subtract %5, %7 : tensor<128x1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %10 = stablehlo.clamp %0, %8, %9 : tensor<128x1xf32>
    return %10 : tensor<128x1xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<128x32x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<128x160x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<128x32x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<128x64x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<128x64x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<128x160x16x16xf32>) -> tensor<128x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
    return %0 : tensor<128x16x16xf32>
  }
  func.func private @wrapped_concatenate_computation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation(%arg0: tensor<32768x64xf32>) -> tensor<32768x32xf32> {
    %0 = stablehlo.slice %arg0 [0:32768, 0:32] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    return %0 : tensor<32768x32xf32>
  }
  func.func private @wrapped_slice_computation.1(%arg0: tensor<32768x64xf32>) -> tensor<32768x32xf32> {
    %0 = stablehlo.slice %arg0 [0:32768, 32:64] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    return %0 : tensor<32768x32xf32>
  }
  func.func private @wrapped_concatenate_computation.1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation.2(%arg0: tensor<32768x64xf32>) -> tensor<32768x32xf32> {
    %0 = stablehlo.slice %arg0 [0:32768, 0:32] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    return %0 : tensor<32768x32xf32>
  }
  func.func private @wrapped_slice_computation.3(%arg0: tensor<32768x64xf32>) -> tensor<32768x32xf32> {
    %0 = stablehlo.slice %arg0 [0:32768, 32:64] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    return %0 : tensor<32768x32xf32>
  }
  func.func private @wrapped_concatenate_computation.2(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func private @wrapped_concatenate_computation.3(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func @main(%arg0: tensor<32x3x7x7xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x3x3xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<160x64x3x3xf32>, %arg9: tensor<160xf32>, %arg10: tensor<160xf32>, %arg11: tensor<160xf32>, %arg12: tensor<256x160x3x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32x32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<32x32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x8x8xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32x32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<128x32xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128x1x3x3xf32>, %arg35: tensor<128xf32>, %arg36: tensor<32x128xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32x32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<32x32xf32>, %arg45: tensor<32xf32>, %arg46: tensor<32x32x8x8xf32>, %arg47: tensor<32xf32>, %arg48: tensor<32xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32x32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<32xf32>, %arg54: tensor<128x32xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x1x3x3xf32>, %arg57: tensor<128xf32>, %arg58: tensor<32x128xf32>, %arg59: tensor<32xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<64x64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64x64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64x64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64x64x4x4xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64x64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<256x64xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<64x256xf32>, %arg81: tensor<64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64xf32>, %arg84: tensor<64x64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64x64xf32>, %arg87: tensor<64xf32>, %arg88: tensor<64x64xf32>, %arg89: tensor<64xf32>, %arg90: tensor<64x64x4x4xf32>, %arg91: tensor<64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64x64xf32>, %arg95: tensor<64xf32>, %arg96: tensor<64xf32>, %arg97: tensor<64xf32>, %arg98: tensor<256x64xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<256xf32>, %arg102: tensor<64x256xf32>, %arg103: tensor<64xf32>, %arg104: tensor<160xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160x160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160x160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<640x160xf32>, %arg121: tensor<640xf32>, %arg122: tensor<640x1x3x3xf32>, %arg123: tensor<640xf32>, %arg124: tensor<160x640xf32>, %arg125: tensor<160xf32>, %arg126: tensor<160xf32>, %arg127: tensor<160xf32>, %arg128: tensor<160x160xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160x160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<640x160xf32>, %arg143: tensor<640xf32>, %arg144: tensor<640x1x3x3xf32>, %arg145: tensor<640xf32>, %arg146: tensor<160x640xf32>, %arg147: tensor<160xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256x256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256x256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256x256xf32>, %arg155: tensor<256xf32>, %arg156: tensor<256x256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024x256xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1x3x3xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256x256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256x256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024x256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024x1x3x3xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<32xf32>, %arg185: tensor<32xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<160xf32>, %arg189: tensor<160xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256x32xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256x64xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256x160xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256x256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256x1024x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<150x256x1x1xf32>, %arg204: tensor<150xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<128x3x512x512xf32>) -> tensor<128x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %302 = stablehlo.add %arg208, %301 : tensor<256xf32>
      %303 = stablehlo.rsqrt %302 : tensor<256xf32>
      mhlo.return %303 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.iota dim = 0 : tensor<128xi32>
      %304 = stablehlo.convert %303 : (tensor<128xi32>) -> tensor<128xf32>
      %305 = stablehlo.maximum %302, %304 : tensor<128xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %307 = stablehlo.convert %306 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %308 = stablehlo.convert %307 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %309 = stablehlo.subtract %306, %308 : tensor<128x1xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %311 = stablehlo.clamp %301, %309, %310 : tensor<128x1xf32>
      mhlo.return %311 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.iota dim = 0 : tensor<128xi32>
      %304 = stablehlo.convert %303 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %307 = stablehlo.multiply %306, %305 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.add %307, %308 : tensor<128xf32>
      %310 = stablehlo.maximum %302, %309 : tensor<128xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %312 = stablehlo.convert %311 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %313 = stablehlo.convert %312 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %314 = stablehlo.subtract %311, %313 : tensor<128x1xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %316 = stablehlo.clamp %301, %314, %315 : tensor<128x1xf32>
      mhlo.return %316 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %3 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.iota dim = 0 : tensor<128xi32>
      %304 = stablehlo.convert %303 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.add %308, %309 : tensor<128xf32>
      %311 = stablehlo.maximum %302, %310 : tensor<128xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %313 = stablehlo.convert %312 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %314 = stablehlo.convert %313 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %315 = stablehlo.subtract %312, %314 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %317 = stablehlo.clamp %301, %315, %316 : tensor<128x1xf32>
      mhlo.return %317 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %4 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.iota dim = 0 : tensor<128xi32>
      %304 = stablehlo.convert %303 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.add %308, %309 : tensor<128xf32>
      %311 = stablehlo.maximum %302, %310 : tensor<128xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %313 = stablehlo.convert %312 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %314 = stablehlo.convert %313 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %315 = stablehlo.subtract %312, %314 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %317 = stablehlo.clamp %301, %315, %316 : tensor<128x1xf32>
      mhlo.return %317 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %5 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg207, %arg0, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<128x3x512x512xf32>, tensor<32x3x7x7xf32>, tensor<32xf32>) -> tuple<tensor<128x32x128x128xf32>, tensor<0xui8>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<128x32x128x128xf32>, tensor<0xui8>>) -> tensor<128x32x128x128xf32>
    %7 = "mhlo.fusion"(%arg3, %arg2, %6) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<128x32x128x128xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<128x128x128xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %308 = stablehlo.subtract %302, %307 : tensor<128x16384x32xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<128x16384x32xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %312 = stablehlo.multiply %310, %311 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x16384xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x16384xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %317 = stablehlo.multiply %308, %316 : tensor<128x16384x32xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<128x16384x32xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %321 = stablehlo.add %319, %320 : tensor<128x16384x32xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %323 = stablehlo.multiply %322, %311 : tensor<128x16384xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<128x16384x32xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<128x16384x32xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %328 = stablehlo.multiply %327, %311 : tensor<128x16384xf32>
      %329 = stablehlo.add %328, %313 : tensor<128x16384xf32>
      %330 = stablehlo.rsqrt %329 : tensor<128x16384xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<128x16384x32xf32>
      mhlo.return %332 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<128x32x128x128xf32>) -> tensor<128x16384x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %9 = "mhlo.fusion"(%8, %arg16, %arg17, %arg18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<2097152x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<2097152x32xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
      mhlo.return %305 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %11 = "mhlo.fusion"(%arg17, %7, %arg16) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<128x16384x32xf32>, %arg210: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x16384x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x16384x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
      mhlo.return %305 : tensor<128x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<128x16384x32xf32>, tensor<32xf32>) -> tensor<128x32x16384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x16384xf32>) -> tensor<128x32x128x128xf32>
    %13 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%12, %arg24, %arg25) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<128x32x16x16xf32>, tensor<0xui8>>
    %14 = stablehlo.get_tuple_element %13[0] : (tuple<tensor<128x32x16x16xf32>, tensor<0xui8>>) -> tensor<128x32x16x16xf32>
    %15 = "mhlo.fusion"(%14) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>) -> tensor<128x16x16xf32>
    %16 = "mhlo.fusion"(%14, %15) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>, %arg209: tensor<128x16x16xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x32x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<128x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x32x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %308 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>, tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %17 = "mhlo.fusion"(%14, %15, %16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x32x256xf32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %308 = stablehlo.multiply %arg210, %307 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %313 = stablehlo.multiply %306, %312 : tensor<128x32x256xf32>
      mhlo.return %313 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>, tensor<128x16x16xf32>, tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %18 = "mhlo.fusion"(%arg20, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %301 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %301 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %20 = "mhlo.fusion"(%17, %arg26, %arg27, %19) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x32x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x32x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x32x256xf32>) -> tensor<128x256x32xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<32768x32xf32>
      %307 = stablehlo.dot %306, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<32768x32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %307 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
    %21 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>):
      %301 = stablehlo.slice %arg208 [0:32768, 0:32] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
      mhlo.return %301 : tensor<32768x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x32xf32>) -> tensor<128x256x32xf32>
    %23 = "mhlo.fusion"(%arg19, %10, %arg21, %22) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<128x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x256x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<128x16384x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<128x256x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x256x32xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.dot_general %302, %305, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16384x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x16384x256xf32>
      mhlo.return %308 : tensor<128x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<128x16384x32xf32>, tensor<32xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x256xf32>
    %24 = "mhlo.fusion"(%23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
      %303 = stablehlo.subtract %arg208, %302 : tensor<128x16384x256xf32>
      %304 = stablehlo.exponential %303 : tensor<128x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
      %307 = stablehlo.divide %304, %306 : tensor<128x16384x256xf32>
      mhlo.return %307 : tensor<128x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x256xf32>) -> tensor<128x16384x256xf32>
    %25 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>):
      %301 = stablehlo.slice %arg208 [0:32768, 32:64] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
      mhlo.return %301 : tensor<32768x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x32xf32>) -> tensor<128x256x32xf32>
    %27 = "mhlo.fusion"(%24, %arg23, %26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<128x256x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<128x256x32xf32>
      %303 = stablehlo.dot_general %arg208, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x256xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
      mhlo.return %303 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x256xf32>, tensor<32xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %29 = "mhlo.fusion"(%28, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
      mhlo.return %301 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    %30 = "mhlo.fusion"(%arg3, %arg2, %6, %29, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<128x32x128x128xf32>, %arg211: tensor<2097152x32xf32>, %arg212: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.add %301, %arg211 : tensor<2097152x32xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %304 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x128x128xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %311 = stablehlo.subtract %305, %310 : tensor<128x16384x32xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<128x16384x32xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x16384xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128x16384xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %320 = stablehlo.multiply %311, %319 : tensor<128x16384x32xf32>
      %321 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x16384x32xf32>
      %323 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x16384x32xf32>
      %325 = stablehlo.add %303, %324 : tensor<128x16384x32xf32>
      %326 = stablehlo.reduce(%325 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x16384xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %329 = stablehlo.subtract %325, %328 : tensor<128x16384x32xf32>
      %330 = stablehlo.multiply %329, %329 : tensor<128x16384x32xf32>
      %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %332 = stablehlo.multiply %331, %314 : tensor<128x16384xf32>
      %333 = stablehlo.add %332, %316 : tensor<128x16384xf32>
      %334 = stablehlo.rsqrt %333 : tensor<128x16384xf32>
      %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %336 = stablehlo.multiply %329, %335 : tensor<128x16384x32xf32>
      mhlo.return %336 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<128x32x128x128xf32>, tensor<2097152x32xf32>, tensor<32xf32>) -> tensor<128x16384x32xf32>
    %31 = mhlo.bitcast %30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %32 = "mhlo.fusion"(%31, %arg30, %arg31, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<2097152x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<2097152x32xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
      mhlo.return %305 : tensor<2097152x128xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
    %33 = "mhlo.fusion"(%32, %arg33) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<2097152x128xf32>, %arg209: tensor<128xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<2097152x128xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<2097152x128xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<128x16384x128xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x16384x128xf32>) -> tensor<128x128x16384xf32>
      mhlo.return %304 : tensor<128x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x128xf32>, tensor<128xf32>) -> tensor<128x128x16384xf32>
    %34 = mhlo.bitcast %33 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x16384xf32>) -> tensor<128x128x128x128xf32>
    %35 = stablehlo.custom_call @__cudnn$convForward(%34, %arg34) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<128x128x128x128xf32>, tensor<0xui8>>
    %36 = stablehlo.get_tuple_element %35[0] : (tuple<tensor<128x128x128x128xf32>, tensor<0xui8>>) -> tensor<128x128x128x128xf32>
    %37 = "mhlo.fusion"(%36, %arg35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<128x128x128x128xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x128x128x128xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x128x128x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
      %306 = stablehlo.negate %303 : tensor<128x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x128x128x128xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
      %310 = stablehlo.abs %309 : tensor<128x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x128x16384xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x128x16384xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x128x16384xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x128x16384xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x128x16384xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x128x16384xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x128x16384xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x128x16384xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %338 = stablehlo.negate %314 : tensor<128x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %341 = stablehlo.exponential %338 : tensor<128x128x16384xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x128x16384xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x128x16384xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x128x16384xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x128x16384xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x128x16384xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x128x16384xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x128x16384xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x128x16384xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x128x16384xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x128x16384xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x128x16384xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x128x16384xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x128x16384xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x128x16384xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x128x16384xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x128x16384xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x128x16384xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x128x16384xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x128x16384xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x128x16384xf32>
      mhlo.return %399 : tensor<128x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x128x128xf32>, tensor<128xf32>) -> tensor<128x128x16384xf32>
    %38 = "mhlo.fusion"(%37, %arg36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x128x16384xf32>) -> tensor<128x16384x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x128xf32>) -> tensor<2097152x128xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
      mhlo.return %303 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x16384xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
    %39 = "mhlo.fusion"(%38, %arg37, %arg3, %arg2, %6, %29, %arg29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<128x32x128x128xf32>, %arg213: tensor<2097152x32xf32>, %arg214: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<2097152x32xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<2097152x32xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %307 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x128x128xf32>) -> tensor<128x32x16384xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1] : (tensor<128x32x16384xf32>) -> tensor<128x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x128x128xf32>, tensor<f32>) -> tensor<128x128x128xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x128x128xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x128xf32>) -> tensor<128x16384xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %314 = stablehlo.subtract %308, %313 : tensor<128x16384x32xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<128x16384x32xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %317 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x16384xf32>
      %321 = stablehlo.rsqrt %320 : tensor<128x16384xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %323 = stablehlo.multiply %314, %322 : tensor<128x16384x32xf32>
      %324 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<128x16384x32xf32>
      %326 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %327 = stablehlo.add %325, %326 : tensor<128x16384x32xf32>
      %328 = stablehlo.add %306, %327 : tensor<128x16384x32xf32>
      %329 = stablehlo.add %303, %328 : tensor<128x16384x32xf32>
      mhlo.return %329 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32x128x128xf32>, tensor<2097152x32xf32>, tensor<32xf32>) -> tensor<128x16384x32xf32>
    %40 = "mhlo.fusion"(%39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x32xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128x16384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<128x16384x32xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<128x16384x32xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x16384xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x16384xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<128x16384x32xf32>
      mhlo.return %313 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x32xf32>) -> tensor<128x16384x32xf32>
    %41 = mhlo.bitcast %40 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %42 = "mhlo.fusion"(%41, %arg38, %arg39, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<2097152x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<2097152x32xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
      mhlo.return %305 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    %43 = mhlo.bitcast %42 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
    %44 = "mhlo.fusion"(%arg39, %40, %arg38) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<128x16384x32xf32>, %arg210: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x16384x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x16384x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
      mhlo.return %305 : tensor<128x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<128x16384x32xf32>, tensor<32xf32>) -> tensor<128x32x16384xf32>
    %45 = mhlo.bitcast %44 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x16384xf32>) -> tensor<128x32x128x128xf32>
    %46 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%45, %arg46, %arg47) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<128x32x16x16xf32>, tensor<0xui8>>
    %47 = stablehlo.get_tuple_element %46[0] : (tuple<tensor<128x32x16x16xf32>, tensor<0xui8>>) -> tensor<128x32x16x16xf32>
    %48 = "mhlo.fusion"(%47) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>) -> tensor<128x16x16xf32>
    %49 = "mhlo.fusion"(%47, %48) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>, %arg209: tensor<128x16x16xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x32x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<128x32x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x32x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %308 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>, tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %50 = "mhlo.fusion"(%47, %48, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x32x16x16xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x32x16x16xf32>) -> tensor<128x32x256xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x32x256xf32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %308 = stablehlo.multiply %arg210, %307 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x32x256xf32>
      %313 = stablehlo.multiply %306, %312 : tensor<128x32x256xf32>
      mhlo.return %313 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x16x16xf32>, tensor<128x16x16xf32>, tensor<128x256xf32>) -> tensor<128x32x256xf32>
    %51 = "mhlo.fusion"(%arg42, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %301 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %301 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %53 = "mhlo.fusion"(%50, %arg48, %arg49, %52) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x32x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x32x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<128x32x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x32x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x32x256xf32>) -> tensor<128x256x32xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<32768x32xf32>
      %307 = stablehlo.dot %306, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<32768x32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %307 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x32x256xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<32768x64xf32>
    %54 = "mhlo.fusion"(%53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>):
      %301 = stablehlo.slice %arg208 [0:32768, 0:32] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
      mhlo.return %301 : tensor<32768x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    %55 = mhlo.bitcast %54 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x32xf32>) -> tensor<128x256x32xf32>
    %56 = "mhlo.fusion"(%arg41, %43, %arg43, %55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<128x16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x256x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<128x16384x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
      %304 = stablehlo.add %303, %arg211 : tensor<128x256x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x256x32xf32>) -> tensor<128x32x256xf32>
      %306 = stablehlo.dot_general %302, %305, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16384x256xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x16384x256xf32>
      mhlo.return %308 : tensor<128x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<128x16384x32xf32>, tensor<32xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x256xf32>
    %57 = "mhlo.fusion"(%56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
      %303 = stablehlo.subtract %arg208, %302 : tensor<128x16384x256xf32>
      %304 = stablehlo.exponential %303 : tensor<128x16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x256xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x256xf32>
      %307 = stablehlo.divide %304, %306 : tensor<128x16384x256xf32>
      mhlo.return %307 : tensor<128x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x256xf32>) -> tensor<128x16384x256xf32>
    %58 = "mhlo.fusion"(%53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>):
      %301 = stablehlo.slice %arg208 [0:32768, 32:64] : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
      mhlo.return %301 : tensor<32768x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>) -> tensor<32768x32xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x32xf32>) -> tensor<128x256x32xf32>
    %60 = "mhlo.fusion"(%57, %arg45, %59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<128x256x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x256x32xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<128x256x32xf32>
      %303 = stablehlo.dot_general %arg208, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x256xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
      mhlo.return %303 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x256xf32>, tensor<32xf32>, tensor<128x256x32xf32>) -> tensor<128x16384x32xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %62 = "mhlo.fusion"(%61, %arg50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
      mhlo.return %301 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32x32xf32>) -> tensor<2097152x32xf32>
    %63 = "mhlo.fusion"(%39, %62, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x32xf32>, %arg209: tensor<2097152x32xf32>, %arg210: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<2097152x32xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<128x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128x16384xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<128x16384x32xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<128x16384x32xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x16384xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x16384xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<128x16384x32xf32>
      mhlo.return %317 : tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x32xf32>, tensor<2097152x32xf32>, tensor<32xf32>) -> tensor<128x16384x32xf32>
    %64 = mhlo.bitcast %63 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x32xf32>) -> tensor<2097152x32xf32>
    %65 = "mhlo.fusion"(%64, %arg52, %arg53, %arg54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2097152x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<2097152x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<2097152x32xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
      mhlo.return %305 : tensor<2097152x128xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<2097152x128xf32>
    %66 = "mhlo.fusion"(%65, %arg55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<2097152x128xf32>, %arg209: tensor<128xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<2097152x128xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<2097152x128xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<128x16384x128xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x16384x128xf32>) -> tensor<128x128x16384xf32>
      mhlo.return %304 : tensor<128x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<2097152x128xf32>, tensor<128xf32>) -> tensor<128x128x16384xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x16384xf32>) -> tensor<128x128x128x128xf32>
    %68 = stablehlo.custom_call @__cudnn$convForward(%67, %arg56) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<128x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<128x128x128x128xf32>, tensor<0xui8>>
    %69 = stablehlo.get_tuple_element %68[0] : (tuple<tensor<128x128x128x128xf32>, tensor<0xui8>>) -> tensor<128x128x128x128xf32>
    %70 = "mhlo.fusion"(%69, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<128x128x128x128xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x128x128x128xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x128x128x128xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
      %306 = stablehlo.negate %303 : tensor<128x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x128x128x128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x128x128x128xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x128x128x128xf32>) -> tensor<128x128x16384xf32>
      %310 = stablehlo.abs %309 : tensor<128x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x128x16384xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x128x16384xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x128x16384xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x128x16384xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x128x16384xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x128x16384xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x128x16384xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x128x16384xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %338 = stablehlo.negate %314 : tensor<128x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %341 = stablehlo.exponential %338 : tensor<128x128x16384xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x128x16384xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x128x16384xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x128x16384xf32>, tensor<128x128x16384xf32>) -> tensor<128x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x128x16384xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x128x16384xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x128x16384xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x128x16384xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x128x16384xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x128x16384xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x128x16384xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x128x16384xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x128x16384xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x128x16384xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x128x16384xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x128x16384xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x128x16384xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x128x16384xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x128x16384xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x128x16384xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x128x16384xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x128x16384xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x128x16384xi1>, tensor<128x128x16384xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x128x16384xf32>
      mhlo.return %399 : tensor<128x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x128x128xf32>, tensor<128xf32>) -> tensor<128x128x16384xf32>
    %71 = "mhlo.fusion"(%70, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x128x16384xf32>, %arg209: tensor<32x128xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x128x16384xf32>) -> tensor<128x16384x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16384x128xf32>) -> tensor<2097152x128xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
      mhlo.return %303 : tensor<2097152x32xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x16384xf32>, tensor<32x128xf32>) -> tensor<2097152x32xf32>
    %72:2 = "mhlo.fusion"(%arg185, %arg184, %71, %arg59, %39, %62, %arg51) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<2097152x32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<128x16384x32xf32>, %arg213: tensor<2097152x32xf32>, %arg214: tensor<32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<2097152x32xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<32xf32>) -> tensor<2097152x32xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<2097152x32xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x32xf32>) -> tensor<128x16384x32xf32>
      %307 = stablehlo.add %306, %arg212 : tensor<128x16384x32xf32>
      %308 = stablehlo.add %303, %307 : tensor<128x16384x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x16384xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<128x16384x32xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<128x16384x32xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x16384x32xf32>, tensor<f32>) -> tensor<128x16384xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<128x16384xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
      %318 = stablehlo.add %316, %317 : tensor<128x16384xf32>
      %319 = stablehlo.rsqrt %318 : tensor<128x16384xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<128x16384xf32>) -> tensor<128x16384x32xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<128x16384x32xf32>
      %322 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %323 = stablehlo.multiply %321, %322 : tensor<128x16384x32xf32>
      %324 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %325 = stablehlo.add %323, %324 : tensor<128x16384x32xf32>
      %326 = stablehlo.transpose %325, dims = [0, 2, 1] : (tensor<128x16384x32xf32>) -> tensor<128x32x16384xf32>
      mhlo.return %326, %321 : tensor<128x32x16384xf32>, tensor<128x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<2097152x32xf32>, tensor<32xf32>, tensor<128x16384x32xf32>, tensor<2097152x32xf32>, tensor<32xf32>) -> (tensor<128x32x16384xf32>, tensor<128x16384x32xf32>)
    %73 = stablehlo.tuple %72#0, %72#1 {xla_shape = "(f32[128,32,16384]{2,1,0}, f32[128,16384,32]{2,1,0})"} : tuple<tensor<128x32x16384xf32>, tensor<128x16384x32xf32>>
    %74 = mhlo.bitcast %72#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x16384xf32>) -> tensor<128x32x128x128xf32>
    %75 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%74, %arg4, %arg5) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<128x32x128x128xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tuple<tensor<128x64x64x64xf32>, tensor<0xui8>>
    %76 = stablehlo.get_tuple_element %75[0] : (tuple<tensor<128x64x64x64xf32>, tensor<0xui8>>) -> tensor<128x64x64x64xf32>
    %77 = "mhlo.fusion"(%arg7, %arg6, %76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<128x64x64x64xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<128x64x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %308 = stablehlo.subtract %302, %307 : tensor<128x4096x64xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<128x4096x64xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %312 = stablehlo.multiply %310, %311 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x4096xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x4096xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %317 = stablehlo.multiply %308, %316 : tensor<128x4096x64xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<128x4096x64xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %321 = stablehlo.add %319, %320 : tensor<128x4096x64xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %323 = stablehlo.multiply %322, %311 : tensor<128x4096xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<128x4096x64xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<128x4096x64xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %328 = stablehlo.multiply %327, %311 : tensor<128x4096xf32>
      %329 = stablehlo.add %328, %313 : tensor<128x4096xf32>
      %330 = stablehlo.rsqrt %329 : tensor<128x4096xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<128x4096x64xf32>
      mhlo.return %332 : tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<128x64x64x64xf32>) -> tensor<128x4096x64xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x64xf32>) -> tensor<524288x64xf32>
    %79 = "mhlo.fusion"(%78, %arg60, %arg61, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<524288x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<524288x64xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
      mhlo.return %305 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    %80 = "mhlo.fusion"(%79, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x2x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x4096x2x32xf32>) -> tensor<128x2x4096x32xf32>
      mhlo.return %304 : tensor<128x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>) -> tensor<128x2x4096x32xf32>
    %81 = mhlo.bitcast %80 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x4096x32xf32>) -> tensor<256x4096x32xf32>
    %82 = "mhlo.fusion"(%arg61, %77, %arg60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<128x4096x64xf32>, %arg210: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x4096x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x4096x64xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
      mhlo.return %305 : tensor<128x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<128x4096x64xf32>, tensor<64xf32>) -> tensor<128x64x4096xf32>
    %83 = mhlo.bitcast %82 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x4096xf32>) -> tensor<128x64x64x64xf32>
    %84 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%83, %arg68, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<128x64x16x16xf32>, tensor<0xui8>>
    %85 = stablehlo.get_tuple_element %84[0] : (tuple<tensor<128x64x16x16xf32>, tensor<0xui8>>) -> tensor<128x64x16x16xf32>
    %86 = "mhlo.fusion"(%85) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>) -> tensor<128x16x16xf32>
    %87 = "mhlo.fusion"(%85, %86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>, %arg209: tensor<128x16x16xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x64x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<128x64x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %308 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>, tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %88 = "mhlo.fusion"(%85, %86, %87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x64x256xf32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %308 = stablehlo.multiply %arg210, %307 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %313 = stablehlo.multiply %306, %312 : tensor<128x64x256xf32>
      mhlo.return %313 : tensor<128x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>, tensor<128x16x16xf32>, tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %89 = "mhlo.fusion"(%88, %arg70, %arg71, %arg64, %arg66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x64x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x64x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x64x256xf32>) -> tensor<128x256x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x64xf32>) -> tensor<32768x64xf32>
      %307 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %309 = stablehlo.dot %306, %308, precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x128xf32>) -> tensor<32768x128xf32>
      mhlo.return %309 : tensor<32768x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<32768x128xf32>
    %90 = "mhlo.fusion"(%89, %arg65) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x128xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 0:64] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 3, 1] : (tensor<128x256x2x32xf32>) -> tensor<128x2x32x256xf32>
      mhlo.return %305 : tensor<128x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x128xf32>, tensor<64xf32>) -> tensor<128x2x32x256xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x32x256xf32>) -> tensor<256x32x256xf32>
    %92 = "mhlo.fusion"(%81, %91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x32xf32>, %arg209: tensor<256x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x4096x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<256x4096x256xf32>
      mhlo.return %303 : tensor<256x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x4096x256xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<128x2x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x2x4096x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x2x4096x256xf32>, tensor<f32>) -> tensor<128x2x4096xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x2x4096x256xf32>
      mhlo.return %309 : tensor<128x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x4096x256xf32>) -> tensor<256x4096x256xf32>
    %95 = "mhlo.fusion"(%89, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x128xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 64:128] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1, 3] : (tensor<128x256x2x32xf32>) -> tensor<128x2x256x32xf32>
      mhlo.return %305 : tensor<128x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x128xf32>, tensor<64xf32>) -> tensor<128x2x256x32xf32>
    %96 = mhlo.bitcast %95 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x256x32xf32>) -> tensor<256x256x32xf32>
    %97 = "mhlo.fusion"(%94, %96) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x256xf32>, %arg209: tensor<256x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
      mhlo.return %301 : tensor<256x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
    %98 = "mhlo.fusion"(%97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x4096x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x32xf32>) -> tensor<128x2x4096x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x2x4096x32xf32>) -> tensor<128x4096x2x32xf32>
      mhlo.return %302 : tensor<128x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x32xf32>) -> tensor<128x4096x2x32xf32>
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x4096x2x32xf32>) -> tensor<524288x64xf32>
    %100 = "mhlo.fusion"(%99, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64x64xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
      mhlo.return %301 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    %101 = "mhlo.fusion"(%arg7, %arg6, %76, %100, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<128x64x64x64xf32>, %arg211: tensor<524288x64xf32>, %arg212: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg211 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %304 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x64x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %311 = stablehlo.subtract %305, %310 : tensor<128x4096x64xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<128x4096x64xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x4096xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128x4096xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %320 = stablehlo.multiply %311, %319 : tensor<128x4096x64xf32>
      %321 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x4096x64xf32>
      %323 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x4096x64xf32>
      %325 = stablehlo.add %303, %324 : tensor<128x4096x64xf32>
      %326 = stablehlo.reduce(%325 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x4096xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %329 = stablehlo.subtract %325, %328 : tensor<128x4096x64xf32>
      %330 = stablehlo.multiply %329, %329 : tensor<128x4096x64xf32>
      %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %332 = stablehlo.multiply %331, %314 : tensor<128x4096xf32>
      %333 = stablehlo.add %332, %316 : tensor<128x4096xf32>
      %334 = stablehlo.rsqrt %333 : tensor<128x4096xf32>
      %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %336 = stablehlo.multiply %329, %335 : tensor<128x4096x64xf32>
      mhlo.return %336 : tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<128x64x64x64xf32>, tensor<524288x64xf32>, tensor<64xf32>) -> tensor<128x4096x64xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x64xf32>) -> tensor<524288x64xf32>
    %103 = "mhlo.fusion"(%102, %arg74, %arg75, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<524288x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<524288x64xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
      mhlo.return %305 : tensor<524288x256xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
    %104 = "mhlo.fusion"(%103, %arg77) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<524288x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<524288x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<524288x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<128x4096x256xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x4096x256xf32>) -> tensor<128x256x4096xf32>
      mhlo.return %304 : tensor<128x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x256xf32>, tensor<256xf32>) -> tensor<128x256x4096xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x4096xf32>) -> tensor<128x256x64x64xf32>
    %106 = stablehlo.custom_call @__cudnn$convForward(%105, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<128x256x64x64xf32>, tensor<0xui8>>
    %107 = stablehlo.get_tuple_element %106[0] : (tuple<tensor<128x256x64x64xf32>, tensor<0xui8>>) -> tensor<128x256x64x64xf32>
    %108 = "mhlo.fusion"(%107, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<128x256x64x64xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x256x64x64xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x256x64x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
      %306 = stablehlo.negate %303 : tensor<128x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x256x64x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
      %310 = stablehlo.abs %309 : tensor<128x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x256x4096xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x256x4096xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x256x4096xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x256x4096xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x256x4096xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x256x4096xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x256x4096xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x256x4096xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %338 = stablehlo.negate %314 : tensor<128x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %341 = stablehlo.exponential %338 : tensor<128x256x4096xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x256x4096xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x256x4096xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x256x4096xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x256x4096xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x256x4096xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x256x4096xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x256x4096xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x256x4096xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x256x4096xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x256x4096xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x256x4096xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x256x4096xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x256x4096xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x256x4096xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x256x4096xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x256x4096xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x256x4096xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x256x4096xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x256x4096xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x256x4096xf32>
      mhlo.return %399 : tensor<128x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x64x64xf32>, tensor<256xf32>) -> tensor<128x256x4096xf32>
    %109 = "mhlo.fusion"(%108, %arg80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x256x4096xf32>) -> tensor<128x4096x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<524288x256xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
      mhlo.return %303 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x4096xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
    %110 = "mhlo.fusion"(%109, %arg81, %arg7, %arg6, %76, %100, %arg73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<128x64x64x64xf32>, %arg213: tensor<524288x64xf32>, %arg214: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<524288x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %307 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x64x64xf32>) -> tensor<128x64x4096xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1] : (tensor<128x64x4096xf32>) -> tensor<128x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x64x64xf32>, tensor<f32>) -> tensor<128x64x64xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x64x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x64x64xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x64xf32>) -> tensor<128x4096xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %314 = stablehlo.subtract %308, %313 : tensor<128x4096x64xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<128x4096x64xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %317 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x4096xf32>
      %321 = stablehlo.rsqrt %320 : tensor<128x4096xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %323 = stablehlo.multiply %314, %322 : tensor<128x4096x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<128x4096x64xf32>
      %326 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %327 = stablehlo.add %325, %326 : tensor<128x4096x64xf32>
      %328 = stablehlo.add %306, %327 : tensor<128x4096x64xf32>
      %329 = stablehlo.add %303, %328 : tensor<128x4096x64xf32>
      mhlo.return %329 : tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x64x64xf32>, tensor<524288x64xf32>, tensor<64xf32>) -> tensor<128x4096x64xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<128x4096x64xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<128x4096x64xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x4096xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x4096xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<128x4096x64xf32>
      mhlo.return %313 : tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x64xf32>) -> tensor<128x4096x64xf32>
    %112 = mhlo.bitcast %111 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x64xf32>) -> tensor<524288x64xf32>
    %113 = "mhlo.fusion"(%112, %arg82, %arg83, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<524288x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<524288x64xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
      mhlo.return %305 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    %114 = "mhlo.fusion"(%113, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x2x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x4096x2x32xf32>) -> tensor<128x2x4096x32xf32>
      mhlo.return %304 : tensor<128x2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>) -> tensor<128x2x4096x32xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x4096x32xf32>) -> tensor<256x4096x32xf32>
    %116 = "mhlo.fusion"(%arg83, %111, %arg82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<128x4096x64xf32>, %arg210: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x4096x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x4096x64xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
      mhlo.return %305 : tensor<128x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<128x4096x64xf32>, tensor<64xf32>) -> tensor<128x64x4096xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x4096xf32>) -> tensor<128x64x64x64xf32>
    %118 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%117, %arg90, %arg91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<128x64x16x16xf32>, tensor<0xui8>>
    %119 = stablehlo.get_tuple_element %118[0] : (tuple<tensor<128x64x16x16xf32>, tensor<0xui8>>) -> tensor<128x64x16x16xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>) -> tensor<128x16x16xf32>
    %121 = "mhlo.fusion"(%119, %120) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>, %arg209: tensor<128x16x16xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x64x256xf32>
      %307 = stablehlo.multiply %306, %306 : tensor<128x64x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.reduce(%307 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<128x64x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %308 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>, tensor<128x16x16xf32>) -> tensor<128x256xf32>
    %122 = "mhlo.fusion"(%119, %120, %121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x64x16x16xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x64x16x16xf32>) -> tensor<128x64x256xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %303 = stablehlo.multiply %arg209, %302 : tensor<128x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %306 = stablehlo.subtract %301, %305 : tensor<128x64x256xf32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %308 = stablehlo.multiply %arg210, %307 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x64x256xf32>
      %313 = stablehlo.multiply %306, %312 : tensor<128x64x256xf32>
      mhlo.return %313 : tensor<128x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x16x16xf32>, tensor<128x16x16xf32>, tensor<128x256xf32>) -> tensor<128x64x256xf32>
    %123 = "mhlo.fusion"(%122, %arg92, %arg93, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x64x256xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x64x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<128x64x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x64x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x64x256xf32>) -> tensor<128x256x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x64xf32>) -> tensor<32768x64xf32>
      %307 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %309 = stablehlo.dot %306, %308, precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x128xf32>) -> tensor<32768x128xf32>
      mhlo.return %309 : tensor<32768x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<32768x128xf32>
    %124 = "mhlo.fusion"(%123, %arg87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x128xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 0:64] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 3, 1] : (tensor<128x256x2x32xf32>) -> tensor<128x2x32x256xf32>
      mhlo.return %305 : tensor<128x2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x128xf32>, tensor<64xf32>) -> tensor<128x2x32x256xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x32x256xf32>) -> tensor<256x32x256xf32>
    %126 = "mhlo.fusion"(%115, %125) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x32xf32>, %arg209: tensor<256x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x4096x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<256x4096x256xf32>
      mhlo.return %303 : tensor<256x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x32xf32>, tensor<256x32x256xf32>) -> tensor<256x4096x256xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<256x4096x256xf32>, tensor<f32>) -> tensor<256x4096xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<128x2x4096xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x2x4096x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x2x4096x256xf32>, tensor<f32>) -> tensor<128x2x4096xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x2x4096xf32>) -> tensor<128x2x4096x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x2x4096x256xf32>
      mhlo.return %309 : tensor<128x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x256xf32>) -> tensor<128x2x4096x256xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x4096x256xf32>) -> tensor<256x4096x256xf32>
    %129 = "mhlo.fusion"(%123, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x128xf32>, %arg209: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 64:128] : (tensor<32768x128xf32>) -> tensor<32768x64xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<128x256x2x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1, 3] : (tensor<128x256x2x32xf32>) -> tensor<128x2x256x32xf32>
      mhlo.return %305 : tensor<128x2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x128xf32>, tensor<64xf32>) -> tensor<128x2x256x32xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x2x256x32xf32>) -> tensor<256x256x32xf32>
    %131 = "mhlo.fusion"(%128, %130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x4096x256xf32>, %arg209: tensor<256x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
      mhlo.return %301 : tensor<256x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x256xf32>, tensor<256x256x32xf32>) -> tensor<256x4096x32xf32>
    %132 = "mhlo.fusion"(%131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x4096x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x4096x32xf32>) -> tensor<128x2x4096x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x2x4096x32xf32>) -> tensor<128x4096x2x32xf32>
      mhlo.return %302 : tensor<128x4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x4096x32xf32>) -> tensor<128x4096x2x32xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x4096x2x32xf32>) -> tensor<524288x64xf32>
    %134 = "mhlo.fusion"(%133, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64x64xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
      mhlo.return %301 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64x64xf32>) -> tensor<524288x64xf32>
    %135 = "mhlo.fusion"(%110, %134, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x64xf32>, %arg209: tensor<524288x64xf32>, %arg210: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<128x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128x4096xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<128x4096x64xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<128x4096x64xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x4096xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x4096xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<128x4096x64xf32>
      mhlo.return %317 : tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x64xf32>, tensor<524288x64xf32>, tensor<64xf32>) -> tensor<128x4096x64xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x64xf32>) -> tensor<524288x64xf32>
    %137 = "mhlo.fusion"(%136, %arg96, %arg97, %arg98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<524288x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<524288x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<524288x64xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
      mhlo.return %305 : tensor<524288x256xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<524288x256xf32>
    %138 = "mhlo.fusion"(%137, %arg99) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<524288x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<524288x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<524288x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<128x4096x256xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x4096x256xf32>) -> tensor<128x256x4096xf32>
      mhlo.return %304 : tensor<128x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<524288x256xf32>, tensor<256xf32>) -> tensor<128x256x4096xf32>
    %139 = mhlo.bitcast %138 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x4096xf32>) -> tensor<128x256x64x64xf32>
    %140 = stablehlo.custom_call @__cudnn$convForward(%139, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<128x256x64x64xf32>, tensor<0xui8>>
    %141 = stablehlo.get_tuple_element %140[0] : (tuple<tensor<128x256x64x64xf32>, tensor<0xui8>>) -> tensor<128x256x64x64xf32>
    %142 = "mhlo.fusion"(%141, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<128x256x64x64xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x256x64x64xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x256x64x64xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
      %306 = stablehlo.negate %303 : tensor<128x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256x64x64xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x256x64x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x64x64xf32>) -> tensor<128x256x4096xf32>
      %310 = stablehlo.abs %309 : tensor<128x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x256x4096xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x256x4096xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x256x4096xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x256x4096xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x256x4096xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x256x4096xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x256x4096xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x256x4096xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %338 = stablehlo.negate %314 : tensor<128x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %341 = stablehlo.exponential %338 : tensor<128x256x4096xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x256x4096xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x256x4096xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x256x4096xf32>, tensor<128x256x4096xf32>) -> tensor<128x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x256x4096xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x256x4096xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x256x4096xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x256x4096xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x256x4096xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x256x4096xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x256x4096xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x256x4096xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x256x4096xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x256x4096xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x256x4096xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x256x4096xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x256x4096xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x256x4096xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x256x4096xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x256x4096xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x256x4096xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x256x4096xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x256x4096xi1>, tensor<128x256x4096xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x256x4096xf32>
      mhlo.return %399 : tensor<128x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x64x64xf32>, tensor<256xf32>) -> tensor<128x256x4096xf32>
    %143 = "mhlo.fusion"(%142, %arg102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x256x4096xf32>, %arg209: tensor<64x256xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x256x4096xf32>) -> tensor<128x4096x256xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<524288x256xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
      mhlo.return %303 : tensor<524288x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x4096xf32>, tensor<64x256xf32>) -> tensor<524288x64xf32>
    %144:2 = "mhlo.fusion"(%arg187, %arg186, %143, %arg103, %110, %134, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<524288x64xf32>, %arg211: tensor<64xf32>, %arg212: tensor<128x4096x64xf32>, %arg213: tensor<524288x64xf32>, %arg214: tensor<64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<524288x64xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<64xf32>) -> tensor<524288x64xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<524288x64xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x64xf32>) -> tensor<128x4096x64xf32>
      %307 = stablehlo.add %306, %arg212 : tensor<128x4096x64xf32>
      %308 = stablehlo.add %303, %307 : tensor<128x4096x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %cst_0 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x4096xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<128x4096x64xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<128x4096x64xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x4096x64xf32>, tensor<f32>) -> tensor<128x4096xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<128x4096xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x4096xf32>
      %318 = stablehlo.add %316, %317 : tensor<128x4096xf32>
      %319 = stablehlo.rsqrt %318 : tensor<128x4096xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<128x4096xf32>) -> tensor<128x4096x64xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<128x4096x64xf32>
      %322 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %323 = stablehlo.multiply %321, %322 : tensor<128x4096x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %325 = stablehlo.add %323, %324 : tensor<128x4096x64xf32>
      %326 = stablehlo.transpose %325, dims = [0, 2, 1] : (tensor<128x4096x64xf32>) -> tensor<128x64x4096xf32>
      mhlo.return %326, %321 : tensor<128x64x4096xf32>, tensor<128x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<524288x64xf32>, tensor<64xf32>, tensor<128x4096x64xf32>, tensor<524288x64xf32>, tensor<64xf32>) -> (tensor<128x64x4096xf32>, tensor<128x4096x64xf32>)
    %145 = stablehlo.tuple %144#0, %144#1 {xla_shape = "(f32[128,64,4096]{2,1,0}, f32[128,4096,64]{2,1,0})"} : tuple<tensor<128x64x4096xf32>, tensor<128x4096x64xf32>>
    %146 = mhlo.bitcast %144#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x64x4096xf32>) -> tensor<128x64x64x64xf32>
    %147 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%146, %arg8, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,160,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<128x64x64x64xf32>, tensor<160x64x3x3xf32>, tensor<160xf32>) -> tuple<tensor<128x160x32x32xf32>, tensor<0xui8>>
    %148 = stablehlo.get_tuple_element %147[0] : (tuple<tensor<128x160x32x32xf32>, tensor<0xui8>>) -> tensor<128x160x32x32xf32>
    %149 = "mhlo.fusion"(%arg11, %arg10, %148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<128x160x32x32xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<128x32x32xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %308 = stablehlo.subtract %302, %307 : tensor<128x1024x160xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<128x1024x160xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %312 = stablehlo.multiply %310, %311 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x1024xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %317 = stablehlo.multiply %308, %316 : tensor<128x1024x160xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<128x1024x160xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %321 = stablehlo.add %319, %320 : tensor<128x1024x160xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %323 = stablehlo.multiply %322, %311 : tensor<128x1024xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<128x1024x160xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<128x1024x160xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %328 = stablehlo.multiply %327, %311 : tensor<128x1024xf32>
      %329 = stablehlo.add %328, %313 : tensor<128x1024xf32>
      %330 = stablehlo.rsqrt %329 : tensor<128x1024xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<128x1024x160xf32>
      mhlo.return %332 : tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<128x160x32x32xf32>) -> tensor<128x1024x160xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x160xf32>) -> tensor<131072x160xf32>
    %151 = "mhlo.fusion"(%150, %arg104, %arg105, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<131072x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<131072x160xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
      mhlo.return %305 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    %152 = "mhlo.fusion"(%151, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x5x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x1024x5x32xf32>) -> tensor<128x5x1024x32xf32>
      mhlo.return %304 : tensor<128x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>) -> tensor<128x5x1024x32xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x1024x32xf32>) -> tensor<640x1024x32xf32>
    %154 = "mhlo.fusion"(%arg105, %149, %arg104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<128x1024x160xf32>, %arg210: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x1024x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x1024x160xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
      mhlo.return %305 : tensor<128x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<128x1024x160xf32>, tensor<160xf32>) -> tensor<128x160x1024xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x160x1024xf32>) -> tensor<128x160x32x32xf32>
    %156 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%155, %arg112, %arg113) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<128x160x16x16xf32>, tensor<0xui8>>
    %157 = stablehlo.get_tuple_element %156[0] : (tuple<tensor<128x160x16x16xf32>, tensor<0xui8>>) -> tensor<128x160x16x16xf32>
    %158 = "mhlo.fusion"(%157) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x160x16x16xf32>) -> tensor<128x16x16xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x16x16xf32>
      mhlo.return %302 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16x16xf32>) -> tensor<128x16x16xf32>
    %160 = "mhlo.fusion"(%159, %157) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x16x16xf32>, %arg209: tensor<128x160x16x16xf32>):
      %301 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
      %302 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.subtract %301, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<128x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %306 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16x16xf32>, tensor<128x160x16x16xf32>) -> tensor<128x256xf32>
    %161 = "mhlo.fusion"(%160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x256xf32>
      %305 = stablehlo.rsqrt %304 : tensor<128x256xf32>
      mhlo.return %305 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256xf32>) -> tensor<128x256xf32>
    %162 = "mhlo.fusion"(%161, %159, %157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x160x16x16xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
      %302 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.subtract %301, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<128x160x256xf32>
      mhlo.return %306 : tensor<128x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256xf32>, tensor<128x16x16xf32>, tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %163 = "mhlo.fusion"(%arg108, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %301 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %301 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %165 = "mhlo.fusion"(%162, %arg114, %arg115, %164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x160x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x160x256xf32>) -> tensor<128x256x160xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x160xf32>) -> tensor<32768x160xf32>
      %307 = stablehlo.dot %306, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<32768x160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
      mhlo.return %307 : tensor<32768x320xf32>
    }) {output_operand_aliasing = []} : (tensor<128x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
    %166 = "mhlo.fusion"(%165, %arg109) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x320xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 0:160] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x160xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 3, 1] : (tensor<128x256x5x32xf32>) -> tensor<128x5x32x256xf32>
      mhlo.return %305 : tensor<128x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x320xf32>, tensor<160xf32>) -> tensor<128x5x32x256xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x32x256xf32>) -> tensor<640x32x256xf32>
    %168 = "mhlo.fusion"(%153, %167) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x32xf32>, %arg209: tensor<640x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640x1024x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<640x1024x256xf32>
      mhlo.return %303 : tensor<640x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<640x1024x256xf32>, tensor<f32>) -> tensor<640x1024xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<128x5x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x5x1024x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x5x1024x256xf32>, tensor<f32>) -> tensor<128x5x1024xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x5x1024x256xf32>
      mhlo.return %309 : tensor<128x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x1024x256xf32>) -> tensor<640x1024x256xf32>
    %171 = "mhlo.fusion"(%165, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x320xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 160:320] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x160xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1, 3] : (tensor<128x256x5x32xf32>) -> tensor<128x5x256x32xf32>
      mhlo.return %305 : tensor<128x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x320xf32>, tensor<160xf32>) -> tensor<128x5x256x32xf32>
    %172 = mhlo.bitcast %171 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x256x32xf32>) -> tensor<640x256x32xf32>
    %173 = "mhlo.fusion"(%170, %172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x256xf32>, %arg209: tensor<640x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
      mhlo.return %301 : tensor<640x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
    %174 = "mhlo.fusion"(%173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1024x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x32xf32>) -> tensor<128x5x1024x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x5x1024x32xf32>) -> tensor<128x1024x5x32xf32>
      mhlo.return %302 : tensor<128x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x32xf32>) -> tensor<128x1024x5x32xf32>
    %175 = mhlo.bitcast %174 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x5x32xf32>) -> tensor<131072x160xf32>
    %176 = "mhlo.fusion"(%175, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160x160xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
      mhlo.return %301 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    %177 = "mhlo.fusion"(%arg11, %arg10, %148, %176, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<128x160x32x32xf32>, %arg211: tensor<131072x160xf32>, %arg212: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg211 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %304 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x32x32xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %311 = stablehlo.subtract %305, %310 : tensor<128x1024x160xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<128x1024x160xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x1024xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128x1024xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %320 = stablehlo.multiply %311, %319 : tensor<128x1024x160xf32>
      %321 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x1024x160xf32>
      %323 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x1024x160xf32>
      %325 = stablehlo.add %303, %324 : tensor<128x1024x160xf32>
      %326 = stablehlo.reduce(%325 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x1024xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %329 = stablehlo.subtract %325, %328 : tensor<128x1024x160xf32>
      %330 = stablehlo.multiply %329, %329 : tensor<128x1024x160xf32>
      %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %332 = stablehlo.multiply %331, %314 : tensor<128x1024xf32>
      %333 = stablehlo.add %332, %316 : tensor<128x1024xf32>
      %334 = stablehlo.rsqrt %333 : tensor<128x1024xf32>
      %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %336 = stablehlo.multiply %329, %335 : tensor<128x1024x160xf32>
      mhlo.return %336 : tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<128x160x32x32xf32>, tensor<131072x160xf32>, tensor<160xf32>) -> tensor<128x1024x160xf32>
    %178 = mhlo.bitcast %177 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x160xf32>) -> tensor<131072x160xf32>
    %179 = "mhlo.fusion"(%178, %arg118, %arg119, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<131072x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<131072x160xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
      mhlo.return %305 : tensor<131072x640xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
    %180 = "mhlo.fusion"(%179, %arg121) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<131072x640xf32>, %arg209: tensor<640xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<131072x640xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<131072x640xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x640xf32>) -> tensor<128x1024x640xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x1024x640xf32>) -> tensor<128x640x1024xf32>
      mhlo.return %304 : tensor<128x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x640xf32>, tensor<640xf32>) -> tensor<128x640x1024xf32>
    %181 = mhlo.bitcast %180 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x640x1024xf32>) -> tensor<128x640x32x32xf32>
    %182 = stablehlo.custom_call @__cudnn$convForward(%181, %arg122) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<128x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<128x640x32x32xf32>, tensor<0xui8>>
    %183 = stablehlo.get_tuple_element %182[0] : (tuple<tensor<128x640x32x32xf32>, tensor<0xui8>>) -> tensor<128x640x32x32xf32>
    %184 = "mhlo.fusion"(%183, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<128x640x32x32xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x640x32x32xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x640x32x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
      %306 = stablehlo.negate %303 : tensor<128x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x640x32x32xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
      %310 = stablehlo.abs %309 : tensor<128x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x640x1024xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x640x1024xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x640x1024xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x640x1024xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x640x1024xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x640x1024xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x640x1024xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x640x1024xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %338 = stablehlo.negate %314 : tensor<128x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %341 = stablehlo.exponential %338 : tensor<128x640x1024xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x640x1024xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x640x1024xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x640x1024xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x640x1024xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x640x1024xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x640x1024xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x640x1024xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x640x1024xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x640x1024xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x640x1024xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x640x1024xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x640x1024xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x640x1024xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x640x1024xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x640x1024xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x640x1024xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x640x1024xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x640x1024xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x640x1024xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x640x1024xf32>
      mhlo.return %399 : tensor<128x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x640x32x32xf32>, tensor<640xf32>) -> tensor<128x640x1024xf32>
    %185 = "mhlo.fusion"(%184, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x640x1024xf32>) -> tensor<128x1024x640xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x640xf32>) -> tensor<131072x640xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x640xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
      mhlo.return %303 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<128x640x1024xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
    %186 = "mhlo.fusion"(%185, %arg125, %arg11, %arg10, %148, %176, %arg117) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<128x160x32x32xf32>, %arg213: tensor<131072x160xf32>, %arg214: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<131072x160xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %307 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x32x32xf32>) -> tensor<128x160x1024xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1] : (tensor<128x160x1024xf32>) -> tensor<128x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x32x32xf32>, tensor<f32>) -> tensor<128x32x32xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x32x32xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x32x32xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x32x32xf32>) -> tensor<128x1024xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %314 = stablehlo.subtract %308, %313 : tensor<128x1024x160xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<128x1024x160xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %317 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x1024xf32>
      %321 = stablehlo.rsqrt %320 : tensor<128x1024xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %323 = stablehlo.multiply %314, %322 : tensor<128x1024x160xf32>
      %324 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<128x1024x160xf32>
      %326 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %327 = stablehlo.add %325, %326 : tensor<128x1024x160xf32>
      %328 = stablehlo.add %306, %327 : tensor<128x1024x160xf32>
      %329 = stablehlo.add %303, %328 : tensor<128x1024x160xf32>
      mhlo.return %329 : tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<128x160x32x32xf32>, tensor<131072x160xf32>, tensor<160xf32>) -> tensor<128x1024x160xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x1024x160xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<128x1024x160xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<128x1024x160xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x1024xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x1024xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<128x1024x160xf32>
      mhlo.return %313 : tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x160xf32>) -> tensor<128x1024x160xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x160xf32>) -> tensor<131072x160xf32>
    %189 = "mhlo.fusion"(%188, %arg126, %arg127, %arg128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<131072x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<131072x160xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
      mhlo.return %305 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    %190 = "mhlo.fusion"(%189, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x5x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x1024x5x32xf32>) -> tensor<128x5x1024x32xf32>
      mhlo.return %304 : tensor<128x5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>) -> tensor<128x5x1024x32xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x1024x32xf32>) -> tensor<640x1024x32xf32>
    %192 = "mhlo.fusion"(%arg127, %187, %arg126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<128x1024x160xf32>, %arg210: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %302 = stablehlo.multiply %arg209, %301 : tensor<128x1024x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x1024x160xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
      mhlo.return %305 : tensor<128x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<128x1024x160xf32>, tensor<160xf32>) -> tensor<128x160x1024xf32>
    %193 = mhlo.bitcast %192 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x160x1024xf32>) -> tensor<128x160x32x32xf32>
    %194 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%193, %arg134, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<128x160x16x16xf32>, tensor<0xui8>>
    %195 = stablehlo.get_tuple_element %194[0] : (tuple<tensor<128x160x16x16xf32>, tensor<0xui8>>) -> tensor<128x160x16x16xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x160x16x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      mhlo.return %301 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x160x16x16xf32>) -> tensor<128x16x16xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x16x16xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x16x16xf32>
      mhlo.return %302 : tensor<128x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16x16xf32>) -> tensor<128x16x16xf32>
    %198 = "mhlo.fusion"(%197, %195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x16x16xf32>, %arg209: tensor<128x160x16x16xf32>):
      %301 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
      %302 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.subtract %301, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.multiply %304, %304 : tensor<128x160x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%305 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x160x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      mhlo.return %306 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16x16xf32>, tensor<128x160x16x16xf32>) -> tensor<128x256xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256xf32>):
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x256xf32>
      %305 = stablehlo.rsqrt %304 : tensor<128x256xf32>
      mhlo.return %305 : tensor<128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256xf32>) -> tensor<128x256xf32>
    %200 = "mhlo.fusion"(%199, %197, %195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256xf32>, %arg209: tensor<128x16x16xf32>, %arg210: tensor<128x160x16x16xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
      %302 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.subtract %301, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.broadcast_in_dim %arg208, dims = [0, 2] : (tensor<128x256xf32>) -> tensor<128x160x256xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<128x160x256xf32>
      mhlo.return %306 : tensor<128x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256xf32>, tensor<128x16x16xf32>, tensor<128x160x16x16xf32>) -> tensor<128x160x256xf32>
    %201 = "mhlo.fusion"(%arg130, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %301 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %301 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %202 = mhlo.bitcast %201 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %203 = "mhlo.fusion"(%200, %arg136, %arg137, %202) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x160x256xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x160x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<128x160x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x160x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x160x256xf32>) -> tensor<128x256x160xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x160xf32>) -> tensor<32768x160xf32>
      %307 = stablehlo.dot %306, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<32768x160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
      mhlo.return %307 : tensor<32768x320xf32>
    }) {output_operand_aliasing = []} : (tensor<128x160x256xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<32768x320xf32>
    %204 = "mhlo.fusion"(%203, %arg131) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x320xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 0:160] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x160xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 3, 1] : (tensor<128x256x5x32xf32>) -> tensor<128x5x32x256xf32>
      mhlo.return %305 : tensor<128x5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x320xf32>, tensor<160xf32>) -> tensor<128x5x32x256xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x32x256xf32>) -> tensor<640x32x256xf32>
    %206 = "mhlo.fusion"(%191, %205) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x32xf32>, %arg209: tensor<640x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<640x1024x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<640x1024x256xf32>
      mhlo.return %303 : tensor<640x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x32xf32>, tensor<640x32x256xf32>) -> tensor<640x1024x256xf32>
    %207 = "mhlo.fusion"(%206) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<640x1024x256xf32>, tensor<f32>) -> tensor<640x1024xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<128x5x1024xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x5x1024x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x5x1024x256xf32>, tensor<f32>) -> tensor<128x5x1024xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x5x1024xf32>) -> tensor<128x5x1024x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x5x1024x256xf32>
      mhlo.return %309 : tensor<128x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x256xf32>) -> tensor<128x5x1024x256xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x1024x256xf32>) -> tensor<640x1024x256xf32>
    %209 = "mhlo.fusion"(%203, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x320xf32>, %arg209: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<32768x160xf32>
      %302 = stablehlo.slice %arg208 [0:32768, 160:320] : (tensor<32768x320xf32>) -> tensor<32768x160xf32>
      %303 = stablehlo.add %301, %302 : tensor<32768x160xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x160xf32>) -> tensor<128x256x5x32xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1, 3] : (tensor<128x256x5x32xf32>) -> tensor<128x5x256x32xf32>
      mhlo.return %305 : tensor<128x5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x320xf32>, tensor<160xf32>) -> tensor<128x5x256x32xf32>
    %210 = mhlo.bitcast %209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x5x256x32xf32>) -> tensor<640x256x32xf32>
    %211 = "mhlo.fusion"(%208, %210) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<640x1024x256xf32>, %arg209: tensor<640x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
      mhlo.return %301 : tensor<640x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x256xf32>, tensor<640x256x32xf32>) -> tensor<640x1024x32xf32>
    %212 = "mhlo.fusion"(%211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1024x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<640x1024x32xf32>) -> tensor<128x5x1024x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x5x1024x32xf32>) -> tensor<128x1024x5x32xf32>
      mhlo.return %302 : tensor<128x1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1024x32xf32>) -> tensor<128x1024x5x32xf32>
    %213 = mhlo.bitcast %212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x5x32xf32>) -> tensor<131072x160xf32>
    %214 = "mhlo.fusion"(%213, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160x160xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
      mhlo.return %301 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160x160xf32>) -> tensor<131072x160xf32>
    %215 = "mhlo.fusion"(%186, %214, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x1024x160xf32>, %arg209: tensor<131072x160xf32>, %arg210: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<128x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128x1024xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<128x1024x160xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<128x1024x160xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x1024xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x1024xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<128x1024x160xf32>
      mhlo.return %317 : tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x160xf32>, tensor<131072x160xf32>, tensor<160xf32>) -> tensor<128x1024x160xf32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x160xf32>) -> tensor<131072x160xf32>
    %217 = "mhlo.fusion"(%216, %arg140, %arg141, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<131072x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<131072x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<131072x160xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
      mhlo.return %305 : tensor<131072x640xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<131072x640xf32>
    %218 = "mhlo.fusion"(%217, %arg143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<131072x640xf32>, %arg209: tensor<640xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<131072x640xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<131072x640xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x640xf32>) -> tensor<128x1024x640xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x1024x640xf32>) -> tensor<128x640x1024xf32>
      mhlo.return %304 : tensor<128x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x640xf32>, tensor<640xf32>) -> tensor<128x640x1024xf32>
    %219 = mhlo.bitcast %218 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x640x1024xf32>) -> tensor<128x640x32x32xf32>
    %220 = stablehlo.custom_call @__cudnn$convForward(%219, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<128x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<128x640x32x32xf32>, tensor<0xui8>>
    %221 = stablehlo.get_tuple_element %220[0] : (tuple<tensor<128x640x32x32xf32>, tensor<0xui8>>) -> tensor<128x640x32x32xf32>
    %222 = "mhlo.fusion"(%221, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<128x640x32x32xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x640x32x32xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x640x32x32xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
      %306 = stablehlo.negate %303 : tensor<128x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x640x32x32xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x640x32x32xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x640x32x32xf32>) -> tensor<128x640x1024xf32>
      %310 = stablehlo.abs %309 : tensor<128x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x640x1024xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x640x1024xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x640x1024xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x640x1024xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x640x1024xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x640x1024xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x640x1024xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x640x1024xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %338 = stablehlo.negate %314 : tensor<128x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %341 = stablehlo.exponential %338 : tensor<128x640x1024xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x640x1024xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x640x1024xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x640x1024xf32>, tensor<128x640x1024xf32>) -> tensor<128x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x640x1024xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x640x1024xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x640x1024xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x640x1024xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x640x1024xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x640x1024xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x640x1024xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x640x1024xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x640x1024xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x640x1024xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x640x1024xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x640x1024xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x640x1024xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x640x1024xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x640x1024xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x640x1024xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x640x1024xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x640x1024xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x640x1024xi1>, tensor<128x640x1024xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x640x1024xf32>
      mhlo.return %399 : tensor<128x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x640x32x32xf32>, tensor<640xf32>) -> tensor<128x640x1024xf32>
    %223 = "mhlo.fusion"(%222, %arg146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x640x1024xf32>, %arg209: tensor<160x640xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x640x1024xf32>) -> tensor<128x1024x640xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x640xf32>) -> tensor<131072x640xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x640xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
      mhlo.return %303 : tensor<131072x160xf32>
    }) {output_operand_aliasing = []} : (tensor<128x640x1024xf32>, tensor<160x640xf32>) -> tensor<131072x160xf32>
    %224:2 = "mhlo.fusion"(%arg189, %arg188, %223, %arg147, %186, %214, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<131072x160xf32>, %arg211: tensor<160xf32>, %arg212: tensor<128x1024x160xf32>, %arg213: tensor<131072x160xf32>, %arg214: tensor<160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %302 = stablehlo.add %301, %arg210 : tensor<131072x160xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<160xf32>) -> tensor<131072x160xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<131072x160xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x160xf32>) -> tensor<128x1024x160xf32>
      %307 = stablehlo.add %306, %arg212 : tensor<128x1024x160xf32>
      %308 = stablehlo.add %303, %307 : tensor<128x1024x160xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %cst_0 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x1024xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<128x1024x160xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<128x1024x160xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x1024x160xf32>, tensor<f32>) -> tensor<128x1024xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<128x1024xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
      %318 = stablehlo.add %316, %317 : tensor<128x1024xf32>
      %319 = stablehlo.rsqrt %318 : tensor<128x1024xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<128x1024xf32>) -> tensor<128x1024x160xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<128x1024x160xf32>
      %322 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %323 = stablehlo.multiply %321, %322 : tensor<128x1024x160xf32>
      %324 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %325 = stablehlo.add %323, %324 : tensor<128x1024x160xf32>
      %326 = stablehlo.transpose %325, dims = [0, 2, 1] : (tensor<128x1024x160xf32>) -> tensor<128x160x1024xf32>
      mhlo.return %326, %321 : tensor<128x160x1024xf32>, tensor<128x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<131072x160xf32>, tensor<160xf32>, tensor<128x1024x160xf32>, tensor<131072x160xf32>, tensor<160xf32>) -> (tensor<128x160x1024xf32>, tensor<128x1024x160xf32>)
    %225 = stablehlo.tuple %224#0, %224#1 {xla_shape = "(f32[128,160,1024]{2,1,0}, f32[128,1024,160]{2,1,0})"} : tuple<tensor<128x160x1024xf32>, tensor<128x1024x160xf32>>
    %226 = mhlo.bitcast %224#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x160x1024xf32>) -> tensor<128x160x32x32xf32>
    %227 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%226, %arg12, %arg13) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x160x32x32xf32>, tensor<256x160x3x3xf32>, tensor<256xf32>) -> tuple<tensor<128x256x16x16xf32>, tensor<0xui8>>
    %228 = stablehlo.get_tuple_element %227[0] : (tuple<tensor<128x256x16x16xf32>, tensor<0xui8>>) -> tensor<128x256x16x16xf32>
    %229 = "mhlo.fusion"(%arg15, %arg14, %228) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<128x256x16x16xf32>):
      %301 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %303 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<128x16x16xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %308 = stablehlo.subtract %302, %307 : tensor<128x256x256xf32>
      %309 = stablehlo.multiply %308, %308 : tensor<128x256x256xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %311 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %312 = stablehlo.multiply %310, %311 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x256xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %317 = stablehlo.multiply %308, %316 : tensor<128x256x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<128x256x256xf32>
      %320 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %321 = stablehlo.add %319, %320 : tensor<128x256x256xf32>
      %322 = stablehlo.reduce(%321 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %323 = stablehlo.multiply %322, %311 : tensor<128x256xf32>
      %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %325 = stablehlo.subtract %321, %324 : tensor<128x256x256xf32>
      %326 = stablehlo.multiply %325, %325 : tensor<128x256x256xf32>
      %327 = stablehlo.reduce(%326 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %328 = stablehlo.multiply %327, %311 : tensor<128x256xf32>
      %329 = stablehlo.add %328, %313 : tensor<128x256xf32>
      %330 = stablehlo.rsqrt %329 : tensor<128x256xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %332 = stablehlo.multiply %325, %331 : tensor<128x256x256xf32>
      mhlo.return %332 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
    %230 = mhlo.bitcast %229 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<32768x256xf32>
    %231 = "mhlo.fusion"(%230, %arg148, %arg149, %arg150) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %232 = "mhlo.fusion"(%231, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
      mhlo.return %304 : tensor<128x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x256x32xf32>
    %233 = mhlo.bitcast %232 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x32xf32>) -> tensor<1024x256x32xf32>
    %234 = "mhlo.fusion"(%230, %arg148, %arg149, %arg152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %235 = "mhlo.fusion"(%234, %arg153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 3, 1] : (tensor<128x256x8x32xf32>) -> tensor<128x8x32x256xf32>
      mhlo.return %304 : tensor<128x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x32x256xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x32x256xf32>) -> tensor<1024x32x256xf32>
    %237 = "mhlo.fusion"(%233, %236) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x32xf32>, %arg209: tensor<1024x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<1024x256x256xf32>
      mhlo.return %303 : tensor<1024x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
    %238 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x256x256xf32>, tensor<f32>) -> tensor<1024x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<128x8x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x8x256x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x8x256x256xf32>, tensor<f32>) -> tensor<128x8x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x8x256x256xf32>
      mhlo.return %309 : tensor<128x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x256xf32>) -> tensor<1024x256x256xf32>
    %240 = "mhlo.fusion"(%230, %arg148, %arg149, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %241 = "mhlo.fusion"(%240, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
      mhlo.return %304 : tensor<128x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x256x32xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x32xf32>) -> tensor<1024x256x32xf32>
    %243 = "mhlo.fusion"(%239, %242) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x256xf32>, %arg209: tensor<1024x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
      mhlo.return %301 : tensor<1024x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x256x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x32xf32>) -> tensor<128x8x256x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x8x256x32xf32>) -> tensor<128x256x8x32xf32>
      mhlo.return %302 : tensor<128x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x32xf32>) -> tensor<128x256x8x32xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x8x32xf32>) -> tensor<32768x256xf32>
    %246 = "mhlo.fusion"(%245, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %301 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %247 = "mhlo.fusion"(%arg15, %arg14, %228, %246, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<128x256x16x16xf32>, %arg211: tensor<32768x256xf32>, %arg212: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg211 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %304 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
      %305 = stablehlo.transpose %304, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %306 = stablehlo.reduce(%arg210 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x16x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %311 = stablehlo.subtract %305, %310 : tensor<128x256x256xf32>
      %312 = stablehlo.multiply %311, %311 : tensor<128x256x256xf32>
      %313 = stablehlo.reduce(%312 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %314 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x256xf32>
      %318 = stablehlo.rsqrt %317 : tensor<128x256xf32>
      %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %320 = stablehlo.multiply %311, %319 : tensor<128x256x256xf32>
      %321 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<128x256x256xf32>
      %323 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %324 = stablehlo.add %322, %323 : tensor<128x256x256xf32>
      %325 = stablehlo.add %303, %324 : tensor<128x256x256xf32>
      %326 = stablehlo.reduce(%325 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x256xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %329 = stablehlo.subtract %325, %328 : tensor<128x256x256xf32>
      %330 = stablehlo.multiply %329, %329 : tensor<128x256x256xf32>
      %331 = stablehlo.reduce(%330 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %332 = stablehlo.multiply %331, %314 : tensor<128x256xf32>
      %333 = stablehlo.add %332, %316 : tensor<128x256xf32>
      %334 = stablehlo.rsqrt %333 : tensor<128x256xf32>
      %335 = stablehlo.broadcast_in_dim %334, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %336 = stablehlo.multiply %329, %335 : tensor<128x256x256xf32>
      mhlo.return %336 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<128x256x16x16xf32>, tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x256x256xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<32768x256xf32>
    %249 = "mhlo.fusion"(%248, %arg158, %arg159, %arg160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
      mhlo.return %305 : tensor<32768x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
    %250 = "mhlo.fusion"(%249, %arg161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x1024xf32>, %arg209: tensor<1024xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<32768x1024xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x1024xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x1024xf32>) -> tensor<128x256x1024xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x256x1024xf32>) -> tensor<128x1024x256xf32>
      mhlo.return %304 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x1024xf32>, tensor<1024xf32>) -> tensor<128x1024x256xf32>
    %251 = mhlo.bitcast %250 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x256xf32>) -> tensor<128x1024x16x16xf32>
    %252 = stablehlo.custom_call @__cudnn$convForward(%251, %arg162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<128x1024x16x16xf32>, tensor<0xui8>>
    %253 = stablehlo.get_tuple_element %252[0] : (tuple<tensor<128x1024x16x16xf32>, tensor<0xui8>>) -> tensor<128x1024x16x16xf32>
    %254 = "mhlo.fusion"(%253, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x16x16xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x1024x16x16xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x1024x16x16xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
      %306 = stablehlo.negate %303 : tensor<128x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x1024x16x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
      %310 = stablehlo.abs %309 : tensor<128x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x1024x256xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x1024x256xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x1024x256xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x1024x256xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x1024x256xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x1024x256xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x1024x256xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x1024x256xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %338 = stablehlo.negate %314 : tensor<128x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %341 = stablehlo.exponential %338 : tensor<128x1024x256xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x1024x256xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x1024x256xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x1024x256xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x1024x256xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x1024x256xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x1024x256xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x1024x256xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x1024x256xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x1024x256xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x1024x256xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x1024x256xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x1024x256xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x1024x256xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x1024x256xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x1024x256xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x1024x256xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x1024x256xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x1024x256xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x1024x256xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x1024x256xf32>
      mhlo.return %399 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x16x16xf32>, tensor<1024xf32>) -> tensor<128x1024x256xf32>
    %255 = "mhlo.fusion"(%254, %arg164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x1024x256xf32>) -> tensor<128x256x1024xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x1024xf32>) -> tensor<32768x1024xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x1024xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
      mhlo.return %303 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x256xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
    %256 = "mhlo.fusion"(%255, %arg165, %arg15, %arg14, %228, %246, %arg157) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128x256x16x16xf32>, %arg213: tensor<32768x256xf32>, %arg214: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %304 = stablehlo.broadcast_in_dim %arg214, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %305 = stablehlo.add %304, %arg213 : tensor<32768x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %307 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x16x16xf32>) -> tensor<128x256x256xf32>
      %308 = stablehlo.transpose %307, dims = [0, 2, 1] : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%arg212 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x256x16x16xf32>, tensor<f32>) -> tensor<128x16x16xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x16x16xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x16x16xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x16x16xf32>) -> tensor<128x256xf32>
      %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %314 = stablehlo.subtract %308, %313 : tensor<128x256x256xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<128x256x256xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %317 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x256xf32>
      %321 = stablehlo.rsqrt %320 : tensor<128x256xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %323 = stablehlo.multiply %314, %322 : tensor<128x256x256xf32>
      %324 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<128x256x256xf32>
      %326 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %327 = stablehlo.add %325, %326 : tensor<128x256x256xf32>
      %328 = stablehlo.add %306, %327 : tensor<128x256x256xf32>
      %329 = stablehlo.add %303, %328 : tensor<128x256x256xf32>
      mhlo.return %329 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x16x16xf32>, tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x256x256xf32>
    %257 = "mhlo.fusion"(%256) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<128x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %305 = stablehlo.subtract %arg208, %304 : tensor<128x256x256xf32>
      %306 = stablehlo.multiply %305, %305 : tensor<128x256x256xf32>
      %307 = stablehlo.reduce(%306 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %308 = stablehlo.multiply %307, %302 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %310 = stablehlo.add %308, %309 : tensor<128x256xf32>
      %311 = stablehlo.rsqrt %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %313 = stablehlo.multiply %305, %312 : tensor<128x256x256xf32>
      mhlo.return %313 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
    %258 = mhlo.bitcast %257 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<32768x256xf32>
    %259 = "mhlo.fusion"(%258, %arg166, %arg167, %arg168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %260 = "mhlo.fusion"(%259, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
      mhlo.return %304 : tensor<128x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x256x32xf32>
    %261 = mhlo.bitcast %260 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x32xf32>) -> tensor<1024x256x32xf32>
    %262 = "mhlo.fusion"(%258, %arg166, %arg167, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %263 = "mhlo.fusion"(%262, %arg171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 3, 1] : (tensor<128x256x8x32xf32>) -> tensor<128x8x32x256xf32>
      mhlo.return %304 : tensor<128x8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x32x256xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x32x256xf32>) -> tensor<1024x32x256xf32>
    %265 = "mhlo.fusion"(%261, %264) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x32xf32>, %arg209: tensor<1024x32x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x256xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<1024x256x256xf32>
      mhlo.return %303 : tensor<1024x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x32xf32>, tensor<1024x32x256xf32>) -> tensor<1024x256x256xf32>
    %266 = "mhlo.fusion"(%265) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x256xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %302 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x256x256xf32>, tensor<f32>) -> tensor<1024x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<128x8x256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
      %305 = stablehlo.subtract %301, %304 : tensor<128x8x256x256xf32>
      %306 = stablehlo.exponential %305 : tensor<128x8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %307 = stablehlo.reduce(%306 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<128x8x256x256xf32>, tensor<f32>) -> tensor<128x8x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<128x8x256xf32>) -> tensor<128x8x256x256xf32>
      %309 = stablehlo.divide %306, %308 : tensor<128x8x256x256xf32>
      mhlo.return %309 : tensor<128x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x256xf32>) -> tensor<128x8x256x256xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x256xf32>) -> tensor<1024x256x256xf32>
    %268 = "mhlo.fusion"(%258, %arg166, %arg167, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %305 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %269 = "mhlo.fusion"(%268, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x8x32xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1, 3] : (tensor<128x256x8x32xf32>) -> tensor<128x8x256x32xf32>
      mhlo.return %304 : tensor<128x8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x8x256x32xf32>
    %270 = mhlo.bitcast %269 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x8x256x32xf32>) -> tensor<1024x256x32xf32>
    %271 = "mhlo.fusion"(%267, %270) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x256x256xf32>, %arg209: tensor<1024x256x32xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
      mhlo.return %301 : tensor<1024x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x256xf32>, tensor<1024x256x32xf32>) -> tensor<1024x256x32xf32>
    %272 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x256x32xf32>):
      %301 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x256x32xf32>) -> tensor<128x8x256x32xf32>
      %302 = stablehlo.transpose %301, dims = [0, 2, 1, 3] : (tensor<128x8x256x32xf32>) -> tensor<128x256x8x32xf32>
      mhlo.return %302 : tensor<128x256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x32xf32>) -> tensor<128x256x8x32xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x256x8x32xf32>) -> tensor<32768x256xf32>
    %274 = "mhlo.fusion"(%273, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256x256xf32>):
      %301 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %301 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %275 = "mhlo.fusion"(%256, %274, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>, %arg209: tensor<32768x256xf32>, %arg210: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg209 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %304 = stablehlo.add %303, %arg208 : tensor<128x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128x256xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %309 = stablehlo.subtract %304, %308 : tensor<128x256x256xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<128x256x256xf32>
      %311 = stablehlo.reduce(%310 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %312 = stablehlo.multiply %311, %306 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %314 = stablehlo.add %312, %313 : tensor<128x256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128x256xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %317 = stablehlo.multiply %309, %316 : tensor<128x256x256xf32>
      mhlo.return %317 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>, tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x256x256xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<32768x256xf32>
    %277 = "mhlo.fusion"(%276, %arg176, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<32768x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<32768x256xf32>
      %305 = stablehlo.dot_general %304, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
      mhlo.return %305 : tensor<32768x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<32768x1024xf32>
    %278 = "mhlo.fusion"(%277, %arg179) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32768x1024xf32>, %arg209: tensor<1024xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<32768x1024xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x1024xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x1024xf32>) -> tensor<128x256x1024xf32>
      %304 = stablehlo.transpose %303, dims = [0, 2, 1] : (tensor<128x256x1024xf32>) -> tensor<128x1024x256xf32>
      mhlo.return %304 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x1024xf32>, tensor<1024xf32>) -> tensor<128x1024x256xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x1024x256xf32>) -> tensor<128x1024x16x16xf32>
    %280 = stablehlo.custom_call @__cudnn$convForward(%279, %arg180) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<128x1024x16x16xf32>, tensor<0xui8>>
    %281 = stablehlo.get_tuple_element %280[0] : (tuple<tensor<128x1024x16x16xf32>, tensor<0xui8>>) -> tensor<128x1024x16x16xf32>
    %282 = "mhlo.fusion"(%281, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
      %302 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024x16x16xf32>
      %303 = stablehlo.add %arg208, %302 : tensor<128x1024x16x16xf32>
      %304 = stablehlo.multiply %301, %303 : tensor<128x1024x16x16xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
      %306 = stablehlo.negate %303 : tensor<128x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1024x16x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128x1024x16x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x1024x16x16xf32>) -> tensor<128x1024x256xf32>
      %310 = stablehlo.abs %309 : tensor<128x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %312 = stablehlo.compare LT, %310, %311 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %314 = stablehlo.multiply %309, %309 : tensor<128x1024x256xf32>
      %315 = stablehlo.multiply %313, %314 : tensor<128x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %317 = stablehlo.add %315, %316 : tensor<128x1024x256xf32>
      %318 = stablehlo.multiply %317, %314 : tensor<128x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %320 = stablehlo.add %318, %319 : tensor<128x1024x256xf32>
      %321 = stablehlo.multiply %320, %314 : tensor<128x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %323 = stablehlo.add %321, %322 : tensor<128x1024x256xf32>
      %324 = stablehlo.multiply %323, %314 : tensor<128x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %325 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %326 = stablehlo.add %324, %325 : tensor<128x1024x256xf32>
      %327 = stablehlo.multiply %326, %314 : tensor<128x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %329 = stablehlo.add %327, %328 : tensor<128x1024x256xf32>
      %330 = stablehlo.multiply %329, %314 : tensor<128x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %332 = stablehlo.add %330, %331 : tensor<128x1024x256xf32>
      %333 = stablehlo.multiply %309, %332 : tensor<128x1024x256xf32>
      %334 = stablehlo.subtract %311, %333 : tensor<128x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %335 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %336 = stablehlo.compare LT, %309, %335 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %338 = stablehlo.negate %314 : tensor<128x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %340 = stablehlo.compare LT, %338, %339 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %341 = stablehlo.exponential %338 : tensor<128x1024x256xf32>
      %342 = stablehlo.divide %311, %310 : tensor<128x1024x256xf32>
      %343 = stablehlo.multiply %341, %342 : tensor<128x1024x256xf32>
      %344 = stablehlo.compare LT, %310, %337 : (tensor<128x1024x256xf32>, tensor<128x1024x256xf32>) -> tensor<128x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %346 = stablehlo.divide %311, %314 : tensor<128x1024x256xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<128x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %349 = stablehlo.add %347, %348 : tensor<128x1024x256xf32>
      %350 = stablehlo.multiply %349, %346 : tensor<128x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %352 = stablehlo.add %350, %351 : tensor<128x1024x256xf32>
      %353 = stablehlo.multiply %352, %346 : tensor<128x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %355 = stablehlo.add %353, %354 : tensor<128x1024x256xf32>
      %356 = stablehlo.multiply %355, %346 : tensor<128x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %358 = stablehlo.add %356, %357 : tensor<128x1024x256xf32>
      %359 = stablehlo.multiply %358, %346 : tensor<128x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %361 = stablehlo.add %359, %360 : tensor<128x1024x256xf32>
      %362 = stablehlo.multiply %361, %346 : tensor<128x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %364 = stablehlo.add %362, %363 : tensor<128x1024x256xf32>
      %365 = stablehlo.multiply %364, %346 : tensor<128x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %367 = stablehlo.add %365, %366 : tensor<128x1024x256xf32>
      %368 = stablehlo.multiply %367, %346 : tensor<128x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %369 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %370 = stablehlo.add %368, %369 : tensor<128x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %372 = stablehlo.multiply %371, %346 : tensor<128x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %374 = stablehlo.add %372, %373 : tensor<128x1024x256xf32>
      %375 = stablehlo.multiply %374, %346 : tensor<128x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %377 = stablehlo.add %375, %376 : tensor<128x1024x256xf32>
      %378 = stablehlo.multiply %377, %346 : tensor<128x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %380 = stablehlo.add %378, %379 : tensor<128x1024x256xf32>
      %381 = stablehlo.multiply %380, %346 : tensor<128x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %383 = stablehlo.add %381, %382 : tensor<128x1024x256xf32>
      %384 = stablehlo.multiply %383, %346 : tensor<128x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %386 = stablehlo.add %384, %385 : tensor<128x1024x256xf32>
      %387 = stablehlo.multiply %386, %346 : tensor<128x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %389 = stablehlo.add %387, %388 : tensor<128x1024x256xf32>
      %390 = stablehlo.multiply %389, %346 : tensor<128x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<128x1024x256xf32>
      %392 = stablehlo.add %390, %391 : tensor<128x1024x256xf32>
      %393 = stablehlo.select %344, %370, %392 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %394 = stablehlo.multiply %343, %393 : tensor<128x1024x256xf32>
      %395 = stablehlo.select %340, %335, %394 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %396 = stablehlo.subtract %337, %395 : tensor<128x1024x256xf32>
      %397 = stablehlo.select %336, %396, %395 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %398 = stablehlo.select %312, %334, %397 : tensor<128x1024x256xi1>, tensor<128x1024x256xf32>
      %399 = stablehlo.multiply %305, %398 : tensor<128x1024x256xf32>
      mhlo.return %399 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x16x16xf32>, tensor<1024xf32>) -> tensor<128x1024x256xf32>
    %283 = "mhlo.fusion"(%282, %arg182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x1024x256xf32>, %arg209: tensor<256x1024xf32>):
      %301 = stablehlo.transpose %arg208, dims = [0, 2, 1] : (tensor<128x1024x256xf32>) -> tensor<128x256x1024xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x1024xf32>) -> tensor<32768x1024xf32>
      %303 = stablehlo.dot_general %302, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x1024xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
      mhlo.return %303 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x256xf32>, tensor<256x1024xf32>) -> tensor<32768x256xf32>
    %284 = "mhlo.fusion"(%283, %arg183, %256, %274, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<128x256x256xf32>, %arg211: tensor<32768x256xf32>, %arg212: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %302 = stablehlo.add %301, %arg208 : tensor<32768x256xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %304 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %305 = stablehlo.add %304, %arg211 : tensor<32768x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<128x256x256xf32>
      %307 = stablehlo.add %306, %arg210 : tensor<128x256x256xf32>
      %308 = stablehlo.add %303, %307 : tensor<128x256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.reduce(%308 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<128x256xf32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %313 = stablehlo.subtract %308, %312 : tensor<128x256x256xf32>
      %314 = stablehlo.multiply %313, %313 : tensor<128x256x256xf32>
      %315 = stablehlo.reduce(%314 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<128x256x256xf32>, tensor<f32>) -> tensor<128x256xf32>
      %316 = stablehlo.multiply %315, %310 : tensor<128x256xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x256xf32>
      %318 = stablehlo.add %316, %317 : tensor<128x256xf32>
      %319 = stablehlo.rsqrt %318 : tensor<128x256xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<128x256xf32>) -> tensor<128x256x256xf32>
      %321 = stablehlo.multiply %313, %320 : tensor<128x256x256xf32>
      mhlo.return %321 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>, tensor<128x256x256xf32>, tensor<32768x256xf32>, tensor<256xf32>) -> tensor<128x256x256xf32>
    %285 = "mhlo.fusion"(%284, %arg190, %arg191, %arg198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x256x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x256x256xf32>
      %305 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<128x256x256xf32>
      %307 = stablehlo.dot_general %304, %306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x256xf32>, tensor<128x256x256xf32>) -> tensor<128x256x256xf32>
      mhlo.return %307 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<128x256x256xf32>
    %286 = "mhlo.fusion"(%285, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<128x256x256xf32>
      %302 = stablehlo.add %arg208, %301 : tensor<128x256x256xf32>
      mhlo.return %302 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>, tensor<256xf32>) -> tensor<128x256x256xf32>
    %287 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.iota dim = 0 : tensor<128xi32>
      %303 = stablehlo.convert %302 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.add %307, %308 : tensor<128xf32>
      %310 = stablehlo.maximum %301, %309 : tensor<128xf32>
      %311 = stablehlo.convert %310 : (tensor<128xf32>) -> tensor<128xi32>
      %312 = stablehlo.convert %311 : (tensor<128xi32>) -> tensor<128xf32>
      %313 = stablehlo.subtract %310, %312 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %315 = stablehlo.clamp %301, %313, %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %288 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.iota dim = 0 : tensor<128xi32>
      %303 = stablehlo.convert %302 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.add %307, %308 : tensor<128xf32>
      %310 = stablehlo.maximum %301, %309 : tensor<128xf32>
      %311 = stablehlo.convert %310 : (tensor<128xf32>) -> tensor<128xi32>
      %312 = stablehlo.convert %311 : (tensor<128xi32>) -> tensor<128xf32>
      %313 = stablehlo.subtract %310, %312 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %315 = stablehlo.clamp %301, %313, %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %289 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.iota dim = 0 : tensor<128xi32>
      %303 = stablehlo.convert %302 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %305 = stablehlo.add %303, %304 : tensor<128xf32>
      %306 = stablehlo.multiply %305, %304 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128xf32>
      %309 = stablehlo.maximum %301, %308 : tensor<128xf32>
      %310 = stablehlo.convert %309 : (tensor<128xf32>) -> tensor<128xi32>
      %311 = stablehlo.convert %310 : (tensor<128xi32>) -> tensor<128xf32>
      %312 = stablehlo.subtract %309, %311 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.clamp %301, %312, %313 : tensor<128xf32>
      mhlo.return %314 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %290 = "mhlo.fusion"(%224#1, %arg188, %arg189, %arg196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<256x160xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x1024x160xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<128x1024x160xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x1024x160xf32>
      %305 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<128x160x256xf32>
      %307 = stablehlo.dot_general %304, %306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1024x160xf32>, tensor<128x160x256xf32>) -> tensor<128x1024x256xf32>
      mhlo.return %307 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<256x160xf32>) -> tensor<128x1024x256xf32>
    %291 = "mhlo.fusion"(%290, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1024x256xf32>, %arg209: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<128x1024x256xf32>
      %302 = stablehlo.add %arg208, %301 : tensor<128x1024x256xf32>
      mhlo.return %302 : tensor<128x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1024x256xf32>, tensor<256xf32>) -> tensor<128x1024x256xf32>
    %292 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %302 = stablehlo.iota dim = 0 : tensor<128xi32>
      %303 = stablehlo.convert %302 : (tensor<128xi32>) -> tensor<128xf32>
      %304 = stablehlo.maximum %301, %303 : tensor<128xf32>
      %305 = stablehlo.convert %304 : (tensor<128xf32>) -> tensor<128xi32>
      %306 = stablehlo.convert %305 : (tensor<128xi32>) -> tensor<128xf32>
      %307 = stablehlo.subtract %304, %306 : tensor<128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %309 = stablehlo.clamp %301, %307, %308 : tensor<128xf32>
      mhlo.return %309 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %293 = "mhlo.fusion"(%72#1, %arg184, %arg185, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x32xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x16384x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<128x16384x32xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x16384x32xf32>
      %305 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<128x32x256xf32>
      %307 = stablehlo.dot_general %304, %306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x16384x32xf32>, tensor<128x32x256xf32>) -> tensor<128x16384x256xf32>
      mhlo.return %307 : tensor<128x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<128x16384x256xf32>
    %294 = "mhlo.fusion"(%144#1, %arg186, %arg187, %arg194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<128x4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %301 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %302 = stablehlo.multiply %arg208, %301 : tensor<128x4096x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<128x4096x64xf32>
      %304 = stablehlo.add %302, %303 : tensor<128x4096x64xf32>
      %305 = stablehlo.transpose %arg211, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<128x64x256xf32>
      %307 = stablehlo.dot_general %304, %306, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x64xf32>, tensor<128x64x256xf32>) -> tensor<128x4096x256xf32>
      mhlo.return %307 : tensor<128x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<128x4096x256xf32>
    %295 = "mhlo.fusion"(%1, %2, %3, %4, %286, %287, %288, %289, %291, %292, %293, %arg193, %294, %arg195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x1xf32>, %arg209: tensor<128x1xf32>, %arg210: tensor<128x1xf32>, %arg211: tensor<128x1xf32>, %arg212: tensor<128x256x256xf32>, %arg213: tensor<128xf32>, %arg214: tensor<128xf32>, %arg215: tensor<128xf32>, %arg216: tensor<128x1024x256xf32>, %arg217: tensor<128xf32>, %arg218: tensor<128x16384x256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<128x4096x256xf32>, %arg221: tensor<256xf32>):
      %301 = mhlo.bitcast %arg212 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,16,16]{1,3,2,0}"} : (tensor<128x256x256xf32>) -> tensor<128x256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %303 = stablehlo.iota dim = 0 : tensor<128xi32>
      %304 = stablehlo.convert %303 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %306 = stablehlo.add %304, %305 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %310 = stablehlo.add %308, %309 : tensor<128xf32>
      %311 = stablehlo.maximum %302, %310 : tensor<128xf32>
      %312 = mhlo.bitcast %311 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %313 = stablehlo.convert %312 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %315 = stablehlo.compare LT, %313, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %316 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %317 = stablehlo.add %313, %316 : tensor<128x1xi32>
      %318 = stablehlo.select %315, %317, %313 : tensor<128x1xi1>, tensor<128x1xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %321 = stablehlo.convert %311 : (tensor<128xf32>) -> tensor<128xi32>
      %322 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %323 = stablehlo.compare LT, %321, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %324 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %325 = stablehlo.add %321, %324 : tensor<128xi32>
      %326 = stablehlo.select %323, %325, %321 : tensor<128xi1>, tensor<128xi32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %328 = stablehlo.concatenate %320, %327, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %329 = mhlo.bitcast %328 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %330 = "stablehlo.gather"(%301, %329) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %331 = mhlo.bitcast %330 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %332 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %333 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %334 = stablehlo.add %321, %333 : tensor<128xi32>
      %335 = stablehlo.minimum %332, %334 : tensor<128xi32>
      %336 = stablehlo.compare LT, %335, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %337 = stablehlo.add %335, %324 : tensor<128xi32>
      %338 = stablehlo.select %336, %337, %335 : tensor<128xi1>, tensor<128xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %340 = stablehlo.concatenate %320, %339, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %342 = "stablehlo.gather"(%301, %341) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %343 = mhlo.bitcast %342 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %344 = stablehlo.subtract %343, %331 : tensor<128x256x128x128xf32>
      %345 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %346 = stablehlo.multiply %344, %345 : tensor<128x256x128x128xf32>
      %347 = stablehlo.add %331, %346 : tensor<128x256x128x128xf32>
      %348 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %349 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %350 = stablehlo.add %313, %349 : tensor<128x1xi32>
      %351 = stablehlo.minimum %348, %350 : tensor<128x1xi32>
      %352 = stablehlo.compare LT, %351, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %353 = stablehlo.add %351, %316 : tensor<128x1xi32>
      %354 = stablehlo.select %352, %353, %351 : tensor<128x1xi1>, tensor<128x1xi32>
      %355 = mhlo.bitcast %354 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %356 = stablehlo.broadcast_in_dim %355, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %357 = stablehlo.concatenate %356, %327, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %358 = mhlo.bitcast %357 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %359 = "stablehlo.gather"(%301, %358) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %360 = mhlo.bitcast %359 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %361 = stablehlo.concatenate %356, %339, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %362 = mhlo.bitcast %361 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %363 = "stablehlo.gather"(%301, %362) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %364 = mhlo.bitcast %363 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %365 = stablehlo.subtract %364, %360 : tensor<128x256x128x128xf32>
      %366 = stablehlo.multiply %365, %345 : tensor<128x256x128x128xf32>
      %367 = stablehlo.add %360, %366 : tensor<128x256x128x128xf32>
      %368 = stablehlo.subtract %367, %347 : tensor<128x256x128x128xf32>
      %369 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %370 = stablehlo.broadcast_in_dim %369, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %371 = stablehlo.multiply %368, %370 : tensor<128x256x128x128xf32>
      %372 = stablehlo.add %347, %371 : tensor<128x256x128x128xf32>
      %373 = mhlo.bitcast %arg216 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,32,32]{1,3,2,0}"} : (tensor<128x1024x256xf32>) -> tensor<128x256x32x32xf32>
      %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %375 = stablehlo.multiply %306, %374 : tensor<128xf32>
      %376 = stablehlo.add %375, %309 : tensor<128xf32>
      %377 = stablehlo.maximum %302, %376 : tensor<128xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %379 = stablehlo.convert %378 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %380 = stablehlo.compare LT, %379, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %381 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %382 = stablehlo.add %379, %381 : tensor<128x1xi32>
      %383 = stablehlo.select %380, %382, %379 : tensor<128x1xi1>, tensor<128x1xi32>
      %384 = mhlo.bitcast %383 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %386 = stablehlo.convert %377 : (tensor<128xf32>) -> tensor<128xi32>
      %387 = stablehlo.compare LT, %386, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %388 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %389 = stablehlo.add %386, %388 : tensor<128xi32>
      %390 = stablehlo.select %387, %389, %386 : tensor<128xi1>, tensor<128xi32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %392 = stablehlo.concatenate %385, %391, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %394 = "stablehlo.gather"(%373, %393) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %c_8 = stablehlo.constant dense<31> : tensor<i32>
      %396 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %397 = stablehlo.add %386, %333 : tensor<128xi32>
      %398 = stablehlo.minimum %396, %397 : tensor<128xi32>
      %399 = stablehlo.compare LT, %398, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %400 = stablehlo.add %398, %388 : tensor<128xi32>
      %401 = stablehlo.select %399, %400, %398 : tensor<128xi1>, tensor<128xi32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %403 = stablehlo.concatenate %385, %402, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %405 = "stablehlo.gather"(%373, %404) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %407 = stablehlo.subtract %406, %395 : tensor<128x256x128x128xf32>
      %408 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<128x256x128x128xf32>
      %410 = stablehlo.add %395, %409 : tensor<128x256x128x128xf32>
      %411 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %412 = stablehlo.add %379, %349 : tensor<128x1xi32>
      %413 = stablehlo.minimum %411, %412 : tensor<128x1xi32>
      %414 = stablehlo.compare LT, %413, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %415 = stablehlo.add %413, %381 : tensor<128x1xi32>
      %416 = stablehlo.select %414, %415, %413 : tensor<128x1xi1>, tensor<128x1xi32>
      %417 = mhlo.bitcast %416 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %419 = stablehlo.concatenate %418, %391, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %421 = "stablehlo.gather"(%373, %420) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %423 = stablehlo.concatenate %418, %402, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %425 = "stablehlo.gather"(%373, %424) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %427 = stablehlo.subtract %426, %422 : tensor<128x256x128x128xf32>
      %428 = stablehlo.multiply %427, %408 : tensor<128x256x128x128xf32>
      %429 = stablehlo.add %422, %428 : tensor<128x256x128x128xf32>
      %430 = stablehlo.subtract %429, %410 : tensor<128x256x128x128xf32>
      %431 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %432 = stablehlo.broadcast_in_dim %431, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %433 = stablehlo.multiply %430, %432 : tensor<128x256x128x128xf32>
      %434 = stablehlo.add %410, %433 : tensor<128x256x128x128xf32>
      %435 = stablehlo.broadcast_in_dim %arg221, dims = [2] : (tensor<256xf32>) -> tensor<128x4096x256xf32>
      %436 = stablehlo.add %arg220, %435 : tensor<128x4096x256xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,64,64]{1,3,2,0}"} : (tensor<128x4096x256xf32>) -> tensor<128x256x64x64xf32>
      %438 = stablehlo.multiply %306, %305 : tensor<128xf32>
      %439 = stablehlo.add %438, %309 : tensor<128xf32>
      %440 = stablehlo.maximum %302, %439 : tensor<128xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %442 = stablehlo.convert %441 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %443 = stablehlo.compare LT, %442, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<64> : tensor<i32>
      %444 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %445 = stablehlo.add %442, %444 : tensor<128x1xi32>
      %446 = stablehlo.select %443, %445, %442 : tensor<128x1xi1>, tensor<128x1xi32>
      %447 = mhlo.bitcast %446 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %448 = stablehlo.broadcast_in_dim %447, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %449 = stablehlo.convert %440 : (tensor<128xf32>) -> tensor<128xi32>
      %450 = stablehlo.compare LT, %449, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %451 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %452 = stablehlo.add %449, %451 : tensor<128xi32>
      %453 = stablehlo.select %450, %452, %449 : tensor<128xi1>, tensor<128xi32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %455 = stablehlo.concatenate %448, %454, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %456 = mhlo.bitcast %455 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %457 = "stablehlo.gather"(%437, %456) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %458 = mhlo.bitcast %457 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %c_10 = stablehlo.constant dense<63> : tensor<i32>
      %459 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %460 = stablehlo.add %449, %333 : tensor<128xi32>
      %461 = stablehlo.minimum %459, %460 : tensor<128xi32>
      %462 = stablehlo.compare LT, %461, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %463 = stablehlo.add %461, %451 : tensor<128xi32>
      %464 = stablehlo.select %462, %463, %461 : tensor<128xi1>, tensor<128xi32>
      %465 = stablehlo.broadcast_in_dim %464, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %466 = stablehlo.concatenate %448, %465, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %468 = "stablehlo.gather"(%437, %467) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %469 = mhlo.bitcast %468 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %470 = stablehlo.subtract %469, %458 : tensor<128x256x128x128xf32>
      %471 = stablehlo.broadcast_in_dim %arg215, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %472 = stablehlo.multiply %470, %471 : tensor<128x256x128x128xf32>
      %473 = stablehlo.add %458, %472 : tensor<128x256x128x128xf32>
      %474 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %475 = stablehlo.add %442, %349 : tensor<128x1xi32>
      %476 = stablehlo.minimum %474, %475 : tensor<128x1xi32>
      %477 = stablehlo.compare LT, %476, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %478 = stablehlo.add %476, %444 : tensor<128x1xi32>
      %479 = stablehlo.select %477, %478, %476 : tensor<128x1xi1>, tensor<128x1xi32>
      %480 = mhlo.bitcast %479 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %481 = stablehlo.broadcast_in_dim %480, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %482 = stablehlo.concatenate %481, %454, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %483 = mhlo.bitcast %482 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %484 = "stablehlo.gather"(%437, %483) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %485 = mhlo.bitcast %484 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %486 = stablehlo.concatenate %481, %465, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %487 = mhlo.bitcast %486 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %488 = "stablehlo.gather"(%437, %487) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %489 = mhlo.bitcast %488 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %490 = stablehlo.subtract %489, %485 : tensor<128x256x128x128xf32>
      %491 = stablehlo.multiply %490, %471 : tensor<128x256x128x128xf32>
      %492 = stablehlo.add %485, %491 : tensor<128x256x128x128xf32>
      %493 = stablehlo.subtract %492, %473 : tensor<128x256x128x128xf32>
      %494 = mhlo.bitcast %arg209 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %495 = stablehlo.broadcast_in_dim %494, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %496 = stablehlo.multiply %493, %495 : tensor<128x256x128x128xf32>
      %497 = stablehlo.add %473, %496 : tensor<128x256x128x128xf32>
      %498 = stablehlo.broadcast_in_dim %arg219, dims = [2] : (tensor<256xf32>) -> tensor<128x16384x256xf32>
      %499 = stablehlo.add %arg218, %498 : tensor<128x16384x256xf32>
      %500 = mhlo.bitcast %499 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[128,256,128,128]{1,3,2,0}"} : (tensor<128x16384x256xf32>) -> tensor<128x256x128x128xf32>
      %501 = stablehlo.maximum %302, %304 : tensor<128xf32>
      %502 = mhlo.bitcast %501 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %503 = stablehlo.convert %502 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %504 = stablehlo.compare LT, %503, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<128> : tensor<i32>
      %505 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %506 = stablehlo.add %503, %505 : tensor<128x1xi32>
      %507 = stablehlo.select %504, %506, %503 : tensor<128x1xi1>, tensor<128x1xi32>
      %508 = mhlo.bitcast %507 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %509 = stablehlo.broadcast_in_dim %508, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %510 = stablehlo.convert %501 : (tensor<128xf32>) -> tensor<128xi32>
      %511 = stablehlo.compare LT, %510, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %512 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %513 = stablehlo.add %510, %512 : tensor<128xi32>
      %514 = stablehlo.select %511, %513, %510 : tensor<128xi1>, tensor<128xi32>
      %515 = stablehlo.broadcast_in_dim %514, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %516 = stablehlo.concatenate %509, %515, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %517 = mhlo.bitcast %516 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %518 = "stablehlo.gather"(%500, %517) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %c_12 = stablehlo.constant dense<127> : tensor<i32>
      %520 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %521 = stablehlo.add %510, %333 : tensor<128xi32>
      %522 = stablehlo.minimum %520, %521 : tensor<128xi32>
      %523 = stablehlo.compare LT, %522, %322 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %524 = stablehlo.add %522, %512 : tensor<128xi32>
      %525 = stablehlo.select %523, %524, %522 : tensor<128xi1>, tensor<128xi32>
      %526 = stablehlo.broadcast_in_dim %525, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %527 = stablehlo.concatenate %509, %526, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %528 = mhlo.bitcast %527 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %529 = "stablehlo.gather"(%500, %528) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %530 = mhlo.bitcast %529 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %531 = stablehlo.subtract %530, %519 : tensor<128x256x128x128xf32>
      %532 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %533 = stablehlo.multiply %531, %532 : tensor<128x256x128x128xf32>
      %534 = stablehlo.add %519, %533 : tensor<128x256x128x128xf32>
      %535 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %536 = stablehlo.add %503, %349 : tensor<128x1xi32>
      %537 = stablehlo.minimum %535, %536 : tensor<128x1xi32>
      %538 = stablehlo.compare LT, %537, %314 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %539 = stablehlo.add %537, %505 : tensor<128x1xi32>
      %540 = stablehlo.select %538, %539, %537 : tensor<128x1xi1>, tensor<128x1xi32>
      %541 = mhlo.bitcast %540 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %542 = stablehlo.broadcast_in_dim %541, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %543 = stablehlo.concatenate %542, %515, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %544 = mhlo.bitcast %543 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %545 = "stablehlo.gather"(%500, %544) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %546 = mhlo.bitcast %545 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %547 = stablehlo.concatenate %542, %526, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %548 = mhlo.bitcast %547 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %549 = "stablehlo.gather"(%500, %548) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 128, 256, 1, 1>}> {result_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>, xla_shape = "f32[16384,128,256,1,1]{0,2,1,4,3}"} : (tensor<128x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x128x256x1x1xf32>
      %550 = mhlo.bitcast %549 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 1, 4, 3]> : tensor<5xindex>} : (tensor<16384x128x256x1x1xf32>) -> tensor<128x256x128x128xf32>
      %551 = stablehlo.subtract %550, %546 : tensor<128x256x128x128xf32>
      %552 = stablehlo.multiply %551, %532 : tensor<128x256x128x128xf32>
      %553 = stablehlo.add %546, %552 : tensor<128x256x128x128xf32>
      %554 = stablehlo.subtract %553, %534 : tensor<128x256x128x128xf32>
      %555 = mhlo.bitcast %arg208 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %556 = stablehlo.broadcast_in_dim %555, dims = [2] : (tensor<128xf32>) -> tensor<128x256x128x128xf32>
      %557 = stablehlo.multiply %554, %556 : tensor<128x256x128x128xf32>
      %558 = stablehlo.add %534, %557 : tensor<128x256x128x128xf32>
      %559 = stablehlo.concatenate %372, %434, %497, %558, dim = 1 : (tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>, tensor<128x256x128x128xf32>) -> tensor<128x1024x128x128xf32>
      mhlo.return %559 : tensor<128x1024x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128x256x256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x1024x256xf32>, tensor<128xf32>, tensor<128x16384x256xf32>, tensor<256xf32>, tensor<128x4096x256xf32>, tensor<256xf32>) -> tensor<128x1024x128x128xf32>
    %296 = stablehlo.custom_call @__cudnn$convForward(%295, %arg200) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,256,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<128x1024x128x128xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<128x256x128x128xf32>, tensor<0xui8>>
    %297 = stablehlo.get_tuple_element %296[0] : (tuple<tensor<128x256x128x128xf32>, tensor<0xui8>>) -> tensor<128x256x128x128xf32>
    %298 = "mhlo.fusion"(%arg202, %arg201, %0, %297, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<128x256x128x128xf32>, %arg212: tensor<256xf32>):
      %301 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
      %302 = stablehlo.subtract %arg211, %301 : tensor<128x256x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<128x256x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<128x256x128x128xf32>
      %307 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<256xf32>) -> tensor<128x256x128x128xf32>
      %308 = stablehlo.add %306, %307 : tensor<128x256x128x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %309 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x256x128x128xf32>
      %310 = stablehlo.maximum %308, %309 : tensor<128x256x128x128xf32>
      mhlo.return %310 : tensor<128x256x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x128x128xf32>, tensor<256xf32>) -> tensor<128x256x128x128xf32>
    %299 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%298, %arg203, %arg204) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[128,150,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<128x256x128x128xf32>, tensor<150x256x1x1xf32>, tensor<150xf32>) -> tuple<tensor<128x150x128x128xf32>, tensor<0xui8>>
    %300 = stablehlo.get_tuple_element %299[0] : (tuple<tensor<128x150x128x128xf32>, tensor<0xui8>>) -> tensor<128x150x128x128xf32>
    return %300 : tensor<128x150x128x128xf32>
  }
}
