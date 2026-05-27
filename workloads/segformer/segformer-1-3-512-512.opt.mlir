module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.68_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    return %4 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot.8_computation(%arg0: tensor<256x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
    %4 = stablehlo.dot %3, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
    return %4 : tensor<256x64xf32>
  }
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<32xf32>, %arg1: tensor<16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x256xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<32xf32>) -> tensor<32x256xf32>
    %3 = stablehlo.add %2, %arg3 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : tensor<32x256xf32>
    %4 = stablehlo.dot %1, %3, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16384x256xf32>
    return %6 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<256x32xf32>
    %2 = stablehlo.dot %arg0, %1, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    return %2 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.73_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.74_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<16384x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    return %4 : tensor<16384x128xf32>
  }
  func.func private @gemm_fusion_dot_general.75_computation(%arg0: tensor<16384x128xf32>, %arg1: tensor<32x128xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.76_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    return %4 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot.9_computation(%arg0: tensor<256x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x64xf32>) -> tensor<256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
    %4 = stablehlo.dot %3, %arg3, precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
    return %4 : tensor<256x64xf32>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<32xf32>, %arg1: tensor<16384x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x256xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<32xf32>) -> tensor<32x256xf32>
    %3 = stablehlo.add %2, %arg3 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : tensor<32x256xf32>
    %4 = stablehlo.dot %1, %3, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16384x256xf32>
    return %6 : tensor<16384x256xf32>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<16384x256xf32>, %arg1: tensor<32xf32>, %arg2: tensor<256x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<256x32xf32>
    %2 = stablehlo.dot %arg0, %1, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    return %2 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.81_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.82_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<128x32xf32>) -> tensor<16384x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    return %4 : tensor<16384x128xf32>
  }
  func.func private @gemm_fusion_dot_general.83_computation(%arg0: tensor<16384x128xf32>, %arg1: tensor<32x128xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    return %0 : tensor<16384x32xf32>
  }
  func.func private @gemm_fusion_dot_general.84_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    return %4 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot.10_computation(%arg0: tensor<256x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64x128xf32>) -> tensor<256x128xf32>
    return %6 : tensor<256x128xf32>
  }
  func.func private @gemm_fusion_dot_general.87_computation(%arg0: tensor<2x32xf32>, %arg1: tensor<4096x2x32xf32>, %arg2: tensor<2x32xf32>, %arg3: tensor<2x32x256xf32>) -> tensor<2x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x32xf32>) -> tensor<4096x2x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<4096x2x32xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4096x2x32xf32>) -> tensor<2x4096x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<2x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<2x4096x256xf32>
    return %7 : tensor<2x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.88_computation(%arg0: tensor<2x4096x256xf32>, %arg1: tensor<2x32xf32>, %arg2: tensor<2x32x256xf32>) -> tensor<4096x2x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<2x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<2x32x256xf32>) -> tensor<2x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<2x4096x32xf32>) -> tensor<4096x2x32xf32>
    return %4 : tensor<4096x2x32xf32>
  }
  func.func private @gemm_fusion_dot_general.89_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    return %0 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot_general.90_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    return %4 : tensor<4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.91_computation(%arg0: tensor<4096x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    return %0 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot_general.92_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    return %4 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot.11_computation(%arg0: tensor<256x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> tensor<256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64x128xf32>) -> tensor<256x128xf32>
    return %6 : tensor<256x128xf32>
  }
  func.func private @gemm_fusion_dot_general.95_computation(%arg0: tensor<2x32xf32>, %arg1: tensor<4096x2x32xf32>, %arg2: tensor<2x32xf32>, %arg3: tensor<2x32x256xf32>) -> tensor<2x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x32xf32>) -> tensor<4096x2x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<4096x2x32xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4096x2x32xf32>) -> tensor<2x4096x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<2x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<2x4096x256xf32>
    return %7 : tensor<2x4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.96_computation(%arg0: tensor<2x4096x256xf32>, %arg1: tensor<2x32xf32>, %arg2: tensor<2x32x256xf32>) -> tensor<4096x2x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<2x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<2x32x256xf32>) -> tensor<2x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<2x4096x32xf32>) -> tensor<4096x2x32xf32>
    return %4 : tensor<4096x2x32xf32>
  }
  func.func private @gemm_fusion_dot_general.97_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    return %0 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot_general.98_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    return %4 : tensor<4096x256xf32>
  }
  func.func private @gemm_fusion_dot_general.99_computation(%arg0: tensor<4096x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<4096x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    return %0 : tensor<4096x64xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    return %4 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot.12_computation(%arg0: tensor<256x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<256x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
    %4 = stablehlo.dot %3, %arg3, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
    return %4 : tensor<256x320xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<5x32xf32>, %arg1: tensor<1024x5x32xf32>, %arg2: tensor<5x32xf32>, %arg3: tensor<5x32x256xf32>) -> tensor<5x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<5x32xf32>) -> tensor<1024x5x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1024x5x32xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<1024x5x32xf32>) -> tensor<5x1024x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<5x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<5x1024x256xf32>
    return %7 : tensor<5x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<5x1024x256xf32>, %arg1: tensor<5x32xf32>, %arg2: tensor<5x32x256xf32>) -> tensor<1024x5x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<5x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<5x32x256xf32>) -> tensor<5x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<5x1024x32xf32>) -> tensor<1024x5x32xf32>
    return %4 : tensor<1024x5x32xf32>
  }
  func.func private @gemm_fusion_dot_general.105_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    return %0 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot_general.106_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<1024x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    return %4 : tensor<1024x640xf32>
  }
  func.func private @gemm_fusion_dot_general.107_computation(%arg0: tensor<1024x640xf32>, %arg1: tensor<160x640xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    return %0 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x160xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    return %4 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot.13_computation(%arg0: tensor<256x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<160x320xf32>) -> tensor<256x320xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
    %4 = stablehlo.dot %3, %arg3, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
    return %4 : tensor<256x320xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<5x32xf32>, %arg1: tensor<1024x5x32xf32>, %arg2: tensor<5x32xf32>, %arg3: tensor<5x32x256xf32>) -> tensor<5x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<5x32xf32>) -> tensor<1024x5x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1024x5x32xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<1024x5x32xf32>) -> tensor<5x1024x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<5x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<5x1024x256xf32>
    return %7 : tensor<5x1024x256xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<5x1024x256xf32>, %arg1: tensor<5x32xf32>, %arg2: tensor<5x32x256xf32>) -> tensor<1024x5x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<5x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<5x32x256xf32>) -> tensor<5x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<5x1024x32xf32>) -> tensor<1024x5x32xf32>
    return %4 : tensor<1024x5x32xf32>
  }
  func.func private @gemm_fusion_dot_general.113_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    return %0 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot_general.114_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<640x160xf32>) -> tensor<1024x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    return %4 : tensor<1024x640xf32>
  }
  func.func private @gemm_fusion_dot_general.115_computation(%arg0: tensor<1024x640xf32>, %arg1: tensor<160x640xf32>) -> tensor<1024x160xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    return %0 : tensor<1024x160xf32>
  }
  func.func private @gemm_fusion_dot.15_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>, %arg4: tensor<256x256xf32>, %arg5: tensor<256x256xf32>) -> tensor<256x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256x768xf32>) -> tensor<256x768xf32>
    return %6 : tensor<256x768xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32x256xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32x256xf32>) -> tensor<8x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<8x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<8x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<8x256x256xf32>
    return %7 : tensor<8x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<8x256x256xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<8x32x256xf32>) -> tensor<256x8x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<8x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<8x256x32xf32>) -> tensor<256x8x32xf32>
    return %4 : tensor<256x8x32xf32>
  }
  func.func private @gemm_fusion_dot_general.121_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.122_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<256x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    return %4 : tensor<256x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.123_computation(%arg0: tensor<256x1024xf32>, %arg1: tensor<256x1024xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @gemm_fusion_dot.17_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>, %arg4: tensor<256x256xf32>, %arg5: tensor<256x256xf32>) -> tensor<256x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %4 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
    %6 = stablehlo.dot %3, %5, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256x768xf32>) -> tensor<256x768xf32>
    return %6 : tensor<256x768xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32x256xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32x256xf32>) -> tensor<8x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<8x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %4 = stablehlo.add %3, %arg3 : tensor<8x32x256xf32>
    %5 = stablehlo.dot_general %2, %4, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<8x256x256xf32>
    return %7 : tensor<8x256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<8x256x256xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<8x32x256xf32>) -> tensor<256x8x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<8x32x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
    %3 = stablehlo.dot_general %arg0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<8x256x32xf32>) -> tensor<256x8x32xf32>
    return %4 : tensor<256x8x32xf32>
  }
  func.func private @gemm_fusion_dot_general.129_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @gemm_fusion_dot_general.130_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256xf32>) -> tensor<256x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    return %4 : tensor<256x1024xf32>
  }
  func.func private @gemm_fusion_dot_general.131_computation(%arg0: tensor<256x1024xf32>, %arg1: tensor<256x1024xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<1024x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<256x160xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<256x160xf32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @gemm_fusion_dot.6_computation(%arg0: tensor<4096x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x64xf32>) -> tensor<4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    return %4 : tensor<4096x256xf32>
  }
  func.func private @gemm_fusion_dot.7_computation(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<256x32xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<256x32xf32>) -> tensor<16384x256xf32>
    return %4 : tensor<16384x256xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.16(%arg0: tensor<8x256x256xf32>) -> tensor<8x256x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<8x256x256xf32>
    %3 = stablehlo.exponential %2 : tensor<8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x256x256xf32>
    return %6 : tensor<8x256x256xf32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.18(%arg0: tensor<8x256x256xf32>) -> tensor<8x256x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<8x256x256xf32>
    %3 = stablehlo.exponential %2 : tensor<8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x256x256xf32>
    return %6 : tensor<8x256x256xf32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.20(%arg0: tensor<5x1024x256xf32>) -> tensor<5x1024x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<5x1024x256xf32>
    %3 = stablehlo.exponential %2 : tensor<5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<5x1024x256xf32>
    return %6 : tensor<5x1024x256xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.22(%arg0: tensor<5x1024x256xf32>) -> tensor<5x1024x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<5x1024x256xf32>
    %3 = stablehlo.exponential %2 : tensor<5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<5x1024x256xf32>
    return %6 : tensor<5x1024x256xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.24(%arg0: tensor<2x4096x256xf32>) -> tensor<2x4096x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<2x4096x256xf32>
    %3 = stablehlo.exponential %2 : tensor<2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<2x4096x256xf32>
    return %6 : tensor<2x4096x256xf32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.26(%arg0: tensor<2x4096x256xf32>) -> tensor<2x4096x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<2x4096x256xf32>
    %3 = stablehlo.exponential %2 : tensor<2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<2x4096x256xf32>
    return %6 : tensor<2x4096x256xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.28(%arg0: tensor<16384x256xf32>) -> tensor<16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<16384x256xf32>
    return %6 : tensor<16384x256xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @triton_softmax_computation.30(%arg0: tensor<16384x256xf32>) -> tensor<16384x256xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
    %2 = stablehlo.subtract %arg0, %1 : tensor<16384x256xf32>
    %3 = stablehlo.exponential %2 : tensor<16384x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
    %6 = stablehlo.divide %3, %5 : tensor<16384x256xf32>
    return %6 : tensor<16384x256xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<1x256x256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x160x1024xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %8 : tensor<1024xf32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024x160xf32>, %arg2: tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<160x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x160x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %6, %4 : tensor<1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x160x1024xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %8 : tensor<1024xf32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.1(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024x160xf32>, %arg2: tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<160x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x160x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %6, %4 : tensor<1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x160x1024xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %8 : tensor<1024xf32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.2(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024x160xf32>, %arg2: tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<160x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x160x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %6, %4 : tensor<1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x160x1024xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %8 : tensor<1024xf32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.3(%arg0: tensor<1x160x32x32xf32>, %arg1: tensor<1024x160xf32>, %arg2: tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<160x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
    %4 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.add %3, %4 : tensor<1x160x1024xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %7, %5 : tensor<1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<1x160x32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<1x160x1024xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x160x1024xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %9 : tensor<1024xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<4096xf32>, %arg1: tensor<1x64x4096xf32>, %arg2: tensor<4096x64xf32>, %arg3: tensor<64xf32>) -> tensor<4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x4096xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<1x64x4096xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<1x64x4096xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %13 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096x64xf32>, %arg2: tensor<64xf32>) -> tensor<4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %6 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x64x4096xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x64x4096xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %8 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.4(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096x64xf32>, %arg2: tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %6, %4 : tensor<4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x64x4096xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x64x4096xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %8 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.5(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096x64xf32>, %arg2: tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %6, %4 : tensor<4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x64x4096xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x64x4096xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %8 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.6(%arg0: tensor<1x64x64x64xf32>, %arg1: tensor<4096x64xf32>, %arg2: tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.add %3, %4 : tensor<1x64x4096xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %7, %5 : tensor<4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<1x64x64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<4096xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<1x64x4096xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x64x4096xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
    return %9 : tensor<4096xf32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<16384xf32>, %arg1: tensor<1x32x16384xf32>, %arg2: tensor<16384x32xf32>, %arg3: tensor<32xf32>) -> tensor<16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<1x32x16384xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<1x32x16384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %13 : tensor<16384xf32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384x32xf32>, %arg2: tensor<32xf32>) -> tensor<16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %6 : tensor<16384xf32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384xf32>) -> tensor<16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x32x16384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x32x16384xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %8 : tensor<16384xf32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.7(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384x32xf32>, %arg2: tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %6, %4 : tensor<16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384xf32>) -> tensor<16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x32x16384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x32x16384xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %8 : tensor<16384xf32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.8(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384x32xf32>, %arg2: tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %6, %4 : tensor<16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384xf32>) -> tensor<16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x32x16384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x32x16384xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %8 : tensor<16384xf32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.9(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32x128x128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<16384x32xf32>, %arg6: tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1x128x128xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %10 = stablehlo.subtract %arg3, %9 : tensor<1x32x128x128xf32>
    %11 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<1x32x128x128xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x32x128x128xf32>
    %17 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x32x128x128xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
    %20 = stablehlo.add %3, %19 : tensor<1x32x16384xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%21 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %22, %20 : tensor<16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32x128x128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128x128xf32>) -> tensor<16384xf32> {
    %0 = mhlo.bitcast %arg5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.subtract %arg3, %5 : tensor<1x32x128x128xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %10 = stablehlo.multiply %6, %9 : tensor<1x32x128x128xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<1x32x128x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x32x128x128xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
    %16 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %17 = stablehlo.multiply %16, %1 : tensor<1x128x128xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %20 = stablehlo.subtract %15, %19 : tensor<1x32x16384xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<1x32x16384xf32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %23 = stablehlo.reduce(%22 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
    return %23 : tensor<16384xf32>
  }
  func.func private @region_0.1.clone.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<1x256x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<1x256x16x16xf32>) -> tensor<16x16xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x256x16x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<1x256x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract(%arg0: tensor<1x160x16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<1x160x16x16xf32>) {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x160x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x160x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9, %6 : tensor<16x16xf32>, tensor<1x160x16x16xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<1x160x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.1(%arg0: tensor<1x160x16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<1x160x16x16xf32>) {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x160x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x160x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9, %6 : tensor<16x16xf32>, tensor<1x160x16x16xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<1x160x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<1x160x32x32xf32>) -> tensor<32x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<1x160x32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x32x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x160x32x32xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x160x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
    return %9 : tensor<32x32xf32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<1x160x32x32xf32>) -> tensor<32x32xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<1x64x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x64x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x64x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<1x64x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1x64x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x64x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x64x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<1x64x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.10(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<1x64x64x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<1x64x64x64xf32>) {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x64x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %6 = stablehlo.subtract %arg3, %5 : tensor<1x64x64x64xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x64x64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x64x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x64x64xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x64x64x64xf32>
    %16 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x64x64x64xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x64x64x64xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
    return %21, %19 : tensor<64x64xf32>, tensor<1x64x64x64xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<1x64x64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x64x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x64x64x64xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x64x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
    return %9 : tensor<64x64xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<1x64x64x64xf32>) -> tensor<64x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<1x32x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x32x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x32x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<1x32x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<1x32x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x32x16x16xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x32x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %9 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1x32x16x16xf32>) -> tensor<16x16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
  func.func private @region_0.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32x128x128xf32>, %arg4: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.subtract %arg3, %5 : tensor<1x32x128x128xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %10 = stablehlo.multiply %6, %9 : tensor<1x32x128x128xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<1x32x128x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x32x128x128xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    return %16 : tensor<128x128xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<1x32x128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x32x128x128xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x32x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    return %9 : tensor<128x128xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<1x32x128x128xf32>) -> tensor<128x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func.func private @fused_add(%arg0: tensor<256x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x256xf32>
    return %2 : tensor<1024x256xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<256x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<1024x256xf32>
    return %2 : tensor<1024x256xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<1024x640xf32>, %arg1: tensor<640xf32>) -> tensor<640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<640xf32>) -> tensor<640x1024xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<640x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<640x1024xf32>
    return %2 : tensor<640x1024xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<1024x640xf32>, %arg1: tensor<640xf32>) -> tensor<640x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<640xf32>) -> tensor<640x1024xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<640x1024xf32>
    %2 = stablehlo.add %0, %1 : tensor<640x1024xf32>
    return %2 : tensor<640x1024xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<4096x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x4096xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<256x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x4096xf32>
    return %2 : tensor<256x4096xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<4096x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x4096xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xf32>) -> tensor<256x4096xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<256x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x4096xf32>
    return %2 : tensor<256x4096xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<16384x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<128x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<128x16384xf32>
    return %2 : tensor<128x16384xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<16384x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<128x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<128x16384xf32>
    return %2 : tensor<128x16384xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x256x256xf32>
    return %4 : tensor<1x256x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x256x256xf32>
    return %4 : tensor<1x256x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x256x256xf32>
    return %4 : tensor<1x256x256xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<1x256x16x16xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
    %4 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<1x256x256xf32>
    return %5 : tensor<1x256x256xf32>
  }
  func.func private @fused_slice(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [0:256, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.1(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [256:512, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.2(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [512:768, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.3(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [0:256, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.4(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [256:512, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.5(%arg0: tensor<256x768xf32>) -> tensor<256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
    %1 = stablehlo.slice %0 [512:768, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_slice.6(%arg0: tensor<256x320xf32>) -> tensor<160x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
    %1 = stablehlo.slice %0 [0:160, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
    return %1 : tensor<160x256xf32>
  }
  func.func private @fused_slice.7(%arg0: tensor<256x320xf32>) -> tensor<160x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
    %1 = stablehlo.slice %0 [160:320, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
    return %1 : tensor<160x256xf32>
  }
  func.func private @fused_slice.8(%arg0: tensor<256x320xf32>) -> tensor<160x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
    %1 = stablehlo.slice %0 [0:160, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
    return %1 : tensor<160x256xf32>
  }
  func.func private @fused_slice.9(%arg0: tensor<256x320xf32>) -> tensor<160x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
    %1 = stablehlo.slice %0 [160:320, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
    return %1 : tensor<160x256xf32>
  }
  func.func private @fused_slice.10(%arg0: tensor<256x128xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %1 = stablehlo.slice %0 [0:64, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @fused_slice.11(%arg0: tensor<256x128xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %1 = stablehlo.slice %0 [64:128, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @fused_slice.12(%arg0: tensor<256x128xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %1 = stablehlo.slice %0 [0:64, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @fused_slice.13(%arg0: tensor<256x128xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %1 = stablehlo.slice %0 [64:128, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<1x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<1x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<1x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<1x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<1x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x1024x256xf32>
    return %98 : tensor<1x1024x256xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<1x1024x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1x1024x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x16x16xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x1024x16x16xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x1024x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
    %5 = stablehlo.negate %2 : tensor<1x1024x16x16xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x1024x16x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
    %9 = stablehlo.abs %8 : tensor<1x1024x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x1024x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x1024x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x1024x256xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x1024x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x1024x256xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x1024x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x1024x256xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x1024x256xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x1024x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x1024x256xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x1024x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x1024x256xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x1024x256xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x1024x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %37 = stablehlo.negate %13 : tensor<1x1024x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %40 = stablehlo.exponential %37 : tensor<1x1024x256xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x1024x256xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x1024x256xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x1024x256xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x1024x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x1024x256xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x1024x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x1024x256xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x1024x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x1024x256xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x1024x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x1024x256xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x1024x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x1024x256xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x1024x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x1024x256xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x1024x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x1024x256xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x1024x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x1024x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x1024x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x1024x256xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x1024x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x1024x256xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x1024x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x1024x256xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x1024x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x1024x256xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x1024x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x1024x256xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x1024x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x1024x256xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x1024x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x1024x256xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x1024x256xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x1024x256xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x1024x256xf32>
    return %98 : tensor<1x1024x256xf32>
  }
  func.func private @fused_subtract(%arg0: tensor<1x256x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<1x256x256xf32>
    return %6 : tensor<1x256x256xf32>
  }
  func.func private @fused_subtract.1(%arg0: tensor<1x256x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<1x256x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x256x16x16xf32>
    %6 = stablehlo.subtract %arg0, %5 : tensor<1x256x16x16xf32>
    return %6 : tensor<1x256x16x16xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<1x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<1x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<1x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<1x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<1x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<1x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<1x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x640x1024xf32>
    return %98 : tensor<1x640x1024xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<1x160x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<1x160x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x16x16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x160x16x16xf32>
    return %9 : tensor<1x160x16x16xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<1x640x32x32xf32>, %arg1: tensor<640xf32>) -> tensor<1x640x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<1x640x32x32xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x640x32x32xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x640x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
    %5 = stablehlo.negate %2 : tensor<1x640x32x32xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x640x32x32xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
    %9 = stablehlo.abs %8 : tensor<1x640x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x640x1024xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x640x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x640x1024xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x640x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x640x1024xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x640x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x640x1024xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x640x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x640x1024xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x640x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x640x1024xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x640x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x640x1024xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x640x1024xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x640x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %37 = stablehlo.negate %13 : tensor<1x640x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %40 = stablehlo.exponential %37 : tensor<1x640x1024xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x640x1024xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x640x1024xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x640x1024xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x640x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x640x1024xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x640x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x640x1024xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x640x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x640x1024xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x640x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x640x1024xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x640x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x640x1024xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x640x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x640x1024xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x640x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x640x1024xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x640x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x640x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x640x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x640x1024xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x640x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x640x1024xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x640x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x640x1024xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x640x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x640x1024xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x640x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x640x1024xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x640x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x640x1024xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x640x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x640x1024xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x640x1024xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x640x1024xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x640x1024xf32>
    return %98 : tensor<1x640x1024xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<1x160x16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<1x160x16x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x16x16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x160x16x16xf32>
    return %9 : tensor<1x160x16x16xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<1x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<1x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<1x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<1x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<1x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x256x4096xf32>
    return %98 : tensor<1x256x4096xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<16x16xf32>, %arg1: tensor<1x64x16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<1x64x16x16xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %6 = stablehlo.subtract %arg1, %5 : tensor<1x64x16x16xf32>
    %7 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x16x16xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x64x16x16xf32>
    return %15 : tensor<1x64x16x16xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<1x256x64x64xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x4096xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x256x64x64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x256x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
    %5 = stablehlo.negate %2 : tensor<1x256x64x64xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x256x64x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
    %9 = stablehlo.abs %8 : tensor<1x256x4096xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x256x4096xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x256x4096xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x256x4096xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x256x4096xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x256x4096xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x256x4096xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x256x4096xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x256x4096xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x256x4096xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x256x4096xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x256x4096xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x256x4096xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x256x4096xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x256x4096xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x256x4096xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %37 = stablehlo.negate %13 : tensor<1x256x4096xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %40 = stablehlo.exponential %37 : tensor<1x256x4096xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x256x4096xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x256x4096xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x256x4096xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x256x4096xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x256x4096xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x256x4096xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x256x4096xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x256x4096xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x256x4096xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x256x4096xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x256x4096xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x256x4096xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x256x4096xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x256x4096xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x256x4096xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x256x4096xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x256x4096xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x256x4096xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x256x4096xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x256x4096xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x256x4096xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x256x4096xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x256x4096xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x256x4096xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x256x4096xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x256x4096xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x256x4096xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x256x4096xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x256x4096xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x256x4096xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x256x4096xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x256x4096xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x256x4096xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x256x4096xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x256x4096xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x256x4096xf32>
    return %98 : tensor<1x256x4096xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<16x16xf32>, %arg1: tensor<1x64x16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<1x64x16x16xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %6 = stablehlo.subtract %arg1, %5 : tensor<1x64x16x16xf32>
    %7 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x16x16xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x64x16x16xf32>
    return %15 : tensor<1x64x16x16xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<1x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<1x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<1x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<1x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<1x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x128x16384xf32>
    return %98 : tensor<1x128x16384xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<16x16xf32>, %arg1: tensor<1x32x16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<1x32x16x16xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %6 = stablehlo.subtract %arg1, %5 : tensor<1x32x16x16xf32>
    %7 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x16x16xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x32x16x16xf32>
    return %15 : tensor<1x32x16x16xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<1x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128x16384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1x128x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
    %5 = stablehlo.negate %2 : tensor<1x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x128x128x128xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
    %9 = stablehlo.abs %8 : tensor<1x128x16384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x128x16384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x128x16384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x128x16384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x128x16384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x128x16384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x128x16384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x128x16384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x128x16384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x128x16384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x128x16384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x128x16384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x128x16384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x128x16384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x128x16384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x128x16384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %37 = stablehlo.negate %13 : tensor<1x128x16384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %40 = stablehlo.exponential %37 : tensor<1x128x16384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x128x16384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x128x16384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x128x16384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x128x16384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x128x16384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x128x16384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x128x16384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x128x16384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x128x16384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x128x16384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x128x16384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x128x16384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x128x16384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x128x16384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x128x16384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x128x16384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x128x16384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x128x16384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x128x16384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x128x16384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x128x16384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x128x16384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x128x16384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x128x16384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x128x16384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x128x16384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x128x16384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x128x16384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x128x16384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x128x16384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x128x16384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x128x16384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x128x16384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x128x16384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x128x16384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x128x16384xf32>
    return %98 : tensor<1x128x16384xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<16x16xf32>, %arg1: tensor<1x32x16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<1x32x16x16xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %6 = stablehlo.subtract %arg1, %5 : tensor<1x32x16x16xf32>
    %7 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x16x16xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x32x16x16xf32>
    return %15 : tensor<1x32x16x16xf32>
  }
  func.func private @fused_maximum(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1x256x128x128xf32>, %arg4: tensor<256xf32>) -> tensor<1x256x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1x256x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x256x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x256x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1x256x128x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x128x128xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<1x256x128x128xf32>
    return %9 : tensor<1x256x128x128xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x256xf32>, %arg4: tensor<128x1xf32>, %arg5: tensor<128x1xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<256x256xf32>, %arg9: tensor<128xf32>, %arg10: tensor<16384x256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<4096x256xf32>, %arg13: tensor<256xf32>) -> tensor<1x1024x128x128xf32> {
    %0 = mhlo.bitcast %arg8 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<256x256xf32>) -> tensor<1x256x16x16xf32>
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
    %29 = "stablehlo.gather"(%0, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
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
    %41 = "stablehlo.gather"(%0, %40) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %43 = stablehlo.subtract %42, %30 : tensor<1x256x128x128xf32>
    %44 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %45 = stablehlo.multiply %43, %44 : tensor<1x256x128x128xf32>
    %46 = stablehlo.add %30, %45 : tensor<1x256x128x128xf32>
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
    %58 = "stablehlo.gather"(%0, %57) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %60 = stablehlo.concatenate %55, %38, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %62 = "stablehlo.gather"(%0, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %64 = stablehlo.subtract %63, %59 : tensor<1x256x128x128xf32>
    %65 = stablehlo.multiply %64, %44 : tensor<1x256x128x128xf32>
    %66 = stablehlo.add %59, %65 : tensor<1x256x128x128xf32>
    %67 = stablehlo.subtract %66, %46 : tensor<1x256x128x128xf32>
    %68 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %70 = stablehlo.multiply %67, %69 : tensor<1x256x128x128xf32>
    %71 = stablehlo.add %46, %70 : tensor<1x256x128x128xf32>
    %72 = mhlo.bitcast %arg3 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,32,32]{1,3,2,0}"} : (tensor<1024x256xf32>) -> tensor<1x256x32x32xf32>
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
    %93 = "stablehlo.gather"(%72, %92) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
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
    %104 = "stablehlo.gather"(%72, %103) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %106 = stablehlo.subtract %105, %94 : tensor<1x256x128x128xf32>
    %107 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %108 = stablehlo.multiply %106, %107 : tensor<1x256x128x128xf32>
    %109 = stablehlo.add %94, %108 : tensor<1x256x128x128xf32>
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
    %120 = "stablehlo.gather"(%72, %119) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %122 = stablehlo.concatenate %117, %101, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %124 = "stablehlo.gather"(%72, %123) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %126 = stablehlo.subtract %125, %121 : tensor<1x256x128x128xf32>
    %127 = stablehlo.multiply %126, %107 : tensor<1x256x128x128xf32>
    %128 = stablehlo.add %121, %127 : tensor<1x256x128x128xf32>
    %129 = stablehlo.subtract %128, %109 : tensor<1x256x128x128xf32>
    %130 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %132 = stablehlo.multiply %129, %131 : tensor<1x256x128x128xf32>
    %133 = stablehlo.add %109, %132 : tensor<1x256x128x128xf32>
    %134 = stablehlo.broadcast_in_dim %arg13, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
    %135 = stablehlo.add %arg12, %134 : tensor<4096x256xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<4096x256xf32>) -> tensor<1x256x64x64xf32>
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
    %156 = "stablehlo.gather"(%136, %155) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
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
    %167 = "stablehlo.gather"(%136, %166) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %168 = mhlo.bitcast %167 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %169 = stablehlo.subtract %168, %157 : tensor<1x256x128x128xf32>
    %170 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<1x256x128x128xf32>
    %172 = stablehlo.add %157, %171 : tensor<1x256x128x128xf32>
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
    %183 = "stablehlo.gather"(%136, %182) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %185 = stablehlo.concatenate %180, %164, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %187 = "stablehlo.gather"(%136, %186) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %189 = stablehlo.subtract %188, %184 : tensor<1x256x128x128xf32>
    %190 = stablehlo.multiply %189, %170 : tensor<1x256x128x128xf32>
    %191 = stablehlo.add %184, %190 : tensor<1x256x128x128xf32>
    %192 = stablehlo.subtract %191, %172 : tensor<1x256x128x128xf32>
    %193 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %195 = stablehlo.multiply %192, %194 : tensor<1x256x128x128xf32>
    %196 = stablehlo.add %172, %195 : tensor<1x256x128x128xf32>
    %197 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %198 = stablehlo.add %arg10, %197 : tensor<16384x256xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,128,128]{1,3,2,0}"} : (tensor<16384x256xf32>) -> tensor<1x256x128x128xf32>
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
    %217 = "stablehlo.gather"(%199, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
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
    %228 = "stablehlo.gather"(%199, %227) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %230 = stablehlo.subtract %229, %218 : tensor<1x256x128x128xf32>
    %231 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %232 = stablehlo.multiply %230, %231 : tensor<1x256x128x128xf32>
    %233 = stablehlo.add %218, %232 : tensor<1x256x128x128xf32>
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
    %244 = "stablehlo.gather"(%199, %243) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %246 = stablehlo.concatenate %241, %225, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %248 = "stablehlo.gather"(%199, %247) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
    %250 = stablehlo.subtract %249, %245 : tensor<1x256x128x128xf32>
    %251 = stablehlo.multiply %250, %231 : tensor<1x256x128x128xf32>
    %252 = stablehlo.add %245, %251 : tensor<1x256x128x128xf32>
    %253 = stablehlo.subtract %252, %233 : tensor<1x256x128x128xf32>
    %254 = mhlo.bitcast %arg5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
    %256 = stablehlo.multiply %253, %255 : tensor<1x256x128x128xf32>
    %257 = stablehlo.add %233, %256 : tensor<1x256x128x128xf32>
    %258 = stablehlo.concatenate %71, %133, %196, %257, dim = 1 : (tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x1024x128x128xf32>
    return %258 : tensor<1x1024x128x128xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384xf32>, %arg2: tensor<16384xf32>) -> tensor<1x32x16384xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x32x16384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x16384xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x16384xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x32x16384xf32>
    return %13 : tensor<1x32x16384xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<1x32x16384xf32>, %arg1: tensor<16384xf32>, %arg2: tensor<16384xf32>) -> tensor<1x32x16384xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x32x16384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x16384xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x16384xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x32x16384xf32>
    return %13 : tensor<1x32x16384xf32>
  }
  func.func private @fused_add_multiply(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<16384xf32>, %arg3: tensor<16384xf32>, %arg4: tensor<1x32x16384xf32>, %arg5: tensor<16384x32xf32>, %arg6: tensor<32xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
    %1 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<1x32x16384xf32>
    %11 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %12 = stablehlo.multiply %11, %6 : tensor<1x16384xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x16384xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x16384xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %18 = stablehlo.multiply %10, %17 : tensor<1x32x16384xf32>
    %19 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<1x32x16384xf32>
    %21 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x32x16384xf32>
    return %22, %18 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @fused_add_multiply.1(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<1x32x16384xf32>, %arg3: tensor<16384xf32>, %arg4: tensor<16384xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>) {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x32x16384xf32>
    %6 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x16384xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x16384xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x16384xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x32x16384xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x32x16384xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x32x16384xf32>
    return %17, %13 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @fused_add_multiply.2(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<1x128x128xf32>, %arg5: tensor<1x32x128x128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<16384xf32>, %arg8: tensor<128x128xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>) {
    %0 = mhlo.bitcast %arg8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x128x128xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %6 = stablehlo.subtract %arg5, %5 : tensor<1x32x128x128xf32>
    %7 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %10 = stablehlo.multiply %6, %9 : tensor<1x32x128x128xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<1x32x128x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x32x128x128xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
    %16 = mhlo.bitcast %arg6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %17 = stablehlo.multiply %16, %1 : tensor<1x128x128xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %20 = stablehlo.subtract %15, %19 : tensor<1x32x16384xf32>
    %21 = mhlo.bitcast %arg7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
    %22 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<1x16384xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x16384xf32>
    %26 = stablehlo.rsqrt %25 : tensor<1x16384xf32>
    %27 = mhlo.bitcast %26 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %29 = stablehlo.multiply %20, %28 : tensor<1x32x16384xf32>
    %30 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<1x32x16384xf32>
    %32 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
    %33 = stablehlo.add %31, %32 : tensor<1x32x16384xf32>
    return %33, %29 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>) -> tensor<1x64x4096xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x64x4096xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x4096xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x64x4096xf32>
    return %13 : tensor<1x64x4096xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1x64x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>) -> tensor<1x64x4096xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x64x4096xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x4096xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x64x4096xf32>
    return %13 : tensor<1x64x4096xf32>
  }
  func.func private @fused_add_multiply.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<4096xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<1x64x4096xf32>, %arg5: tensor<4096x64xf32>, %arg6: tensor<64xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
    %1 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
    %2 = stablehlo.add %0, %1 : tensor<64x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
    %4 = stablehlo.add %3, %arg4 : tensor<1x64x4096xf32>
    %5 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x4096xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<1x64x4096xf32>
    %11 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %12 = stablehlo.multiply %11, %6 : tensor<1x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x4096xf32>
    %15 = stablehlo.rsqrt %14 : tensor<1x4096xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %18 = stablehlo.multiply %10, %17 : tensor<1x64x4096xf32>
    %19 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<1x64x4096xf32>
    %21 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x64x4096xf32>
    return %22, %18 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @fused_add_multiply.4(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x4096xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<4096xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>) {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4096xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x64x4096xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x4096xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x4096xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x64x4096xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x64x4096xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x64x4096xf32>
    return %17, %13 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @fused_add_multiply.5(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64x64x64xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<64x64xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>) {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<1x64x4096xf32>
    %7 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x4096xf32>
    %12 = stablehlo.rsqrt %11 : tensor<1x4096xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x64x4096xf32>
    %16 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x64x4096xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x64x4096xf32>
    return %19, %15 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<1024x256xf32>, %arg1: tensor<256xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1024x256xf32>
    return %1 : tensor<1024x256xf32>
  }
  func.func private @fused_add_multiply.6(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<1x160x1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>) {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x160x1024xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x1024xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x160x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x160x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x160x1024xf32>
    return %17, %13 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @fused_add_multiply.7(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<1x160x1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>) {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x160x1024xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x1024xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x160x1024xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x160x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x160x1024xf32>
    return %17, %13 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @fused_add_multiply.8(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<1x160x32x32xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<32x32xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>) {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
    %1 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<1x160x1024xf32>
    %7 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x1024xf32>
    %12 = stablehlo.rsqrt %11 : tensor<1x1024xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x160x1024xf32>
    %16 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x160x1024xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x160x1024xf32>
    return %19, %15 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<1x160x32x32xf32>, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>) -> tensor<1x160x32x32xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x32x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %6 = stablehlo.subtract %arg2, %5 : tensor<1x160x32x32xf32>
    %7 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<1x32x32xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x32x32xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x32x32xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %15 = stablehlo.multiply %6, %14 : tensor<1x160x32x32xf32>
    %16 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<1x160x32x32xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<1x160x32x32xf32>
    %18 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<160xf32>) -> tensor<1x160x32x32xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x160x32x32xf32>
    return %19 : tensor<1x160x32x32xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1x160x1024xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x1024xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x160x1024xf32>
    return %13 : tensor<1x160x1024xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1x160x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1x160x1024xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x1024xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x160x1024xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x1024xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x1024xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x160x1024xf32>
    return %13 : tensor<1x160x1024xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<1x256x16x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x16xf32>) -> tensor<1x256x16x16xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x16x16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x16x16xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x256x16x16xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x256x16x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256x16x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x256x16x16xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256x16x16xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x256x16x16xf32>
    return %13 : tensor<1x256x16x16xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x256xf32>
    return %8 : tensor<1x256x256xf32>
  }
  func.func private @fused_subtract.2(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x256x256xf32>
    return %5 : tensor<1x256x256xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x256xf32>
    return %8 : tensor<1x256x256xf32>
  }
  func.func private @fused_subtract.3(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x256x256xf32>
    return %5 : tensor<1x256x256xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x256xf32>
    return %8 : tensor<1x256x256xf32>
  }
  func.func private @fused_subtract.4(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x256x256xf32>
    return %5 : tensor<1x256x256xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x256xf32>
    return %8 : tensor<1x256x256xf32>
  }
  func.func private @fused_subtract.5(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x256x256xf32>
    return %5 : tensor<1x256x256xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<1x256x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x256xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x256x256xf32>
    return %8 : tensor<1x256x256xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x128x128xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x128x128xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x128x128xf32>
    return %5 : tensor<1x128x128xf32>
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
  func.func private @fused_rsqrt.1(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @wrapped_concatenate_computation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation(%arg0: tensor<256x64xf32>) -> tensor<256x32xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 0:32] : (tensor<256x64xf32>) -> tensor<256x32xf32>
    return %0 : tensor<256x32xf32>
  }
  func.func private @wrapped_slice_computation.1(%arg0: tensor<256x64xf32>) -> tensor<256x32xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 32:64] : (tensor<256x64xf32>) -> tensor<256x32xf32>
    return %0 : tensor<256x32xf32>
  }
  func.func private @wrapped_concatenate_computation.1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
  }
  func.func private @wrapped_slice_computation.2(%arg0: tensor<256x64xf32>) -> tensor<256x32xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 0:32] : (tensor<256x64xf32>) -> tensor<256x32xf32>
    return %0 : tensor<256x32xf32>
  }
  func.func private @wrapped_slice_computation.3(%arg0: tensor<256x64xf32>) -> tensor<256x32xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 32:64] : (tensor<256x64xf32>) -> tensor<256x32xf32>
    return %0 : tensor<256x32xf32>
  }
  func.func private @wrapped_concatenate_computation.2(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func private @wrapped_concatenate_computation.3(%arg0: tensor<160x160xf32>, %arg1: tensor<160x160xf32>) -> tensor<320x160xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    return %0 : tensor<320x160xf32>
  }
  func.func @main(%arg0: tensor<32x3x7x7xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x3x3xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<160x64x3x3xf32>, %arg9: tensor<160xf32>, %arg10: tensor<160xf32>, %arg11: tensor<160xf32>, %arg12: tensor<256x160x3x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32x32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<32x32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x8x8xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32x32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<128x32xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128x1x3x3xf32>, %arg35: tensor<128xf32>, %arg36: tensor<32x128xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32x32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<32x32xf32>, %arg45: tensor<32xf32>, %arg46: tensor<32x32x8x8xf32>, %arg47: tensor<32xf32>, %arg48: tensor<32xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32x32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<32xf32>, %arg54: tensor<128x32xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x1x3x3xf32>, %arg57: tensor<128xf32>, %arg58: tensor<32x128xf32>, %arg59: tensor<32xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<64x64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64x64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64x64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64x64x4x4xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64x64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<256x64xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<64x256xf32>, %arg81: tensor<64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64xf32>, %arg84: tensor<64x64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64x64xf32>, %arg87: tensor<64xf32>, %arg88: tensor<64x64xf32>, %arg89: tensor<64xf32>, %arg90: tensor<64x64x4x4xf32>, %arg91: tensor<64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64x64xf32>, %arg95: tensor<64xf32>, %arg96: tensor<64xf32>, %arg97: tensor<64xf32>, %arg98: tensor<256x64xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<256xf32>, %arg102: tensor<64x256xf32>, %arg103: tensor<64xf32>, %arg104: tensor<160xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160x160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160x160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<640x160xf32>, %arg121: tensor<640xf32>, %arg122: tensor<640x1x3x3xf32>, %arg123: tensor<640xf32>, %arg124: tensor<160x640xf32>, %arg125: tensor<160xf32>, %arg126: tensor<160xf32>, %arg127: tensor<160xf32>, %arg128: tensor<160x160xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160x160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<640x160xf32>, %arg143: tensor<640xf32>, %arg144: tensor<640x1x3x3xf32>, %arg145: tensor<640xf32>, %arg146: tensor<160x640xf32>, %arg147: tensor<160xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256x256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256x256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256x256xf32>, %arg155: tensor<256xf32>, %arg156: tensor<256x256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024x256xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1x3x3xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256x256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256x256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024x256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024x1x3x3xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<32xf32>, %arg185: tensor<32xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<160xf32>, %arg189: tensor<160xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256x32xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256x64xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256x160xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256x256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256x1024x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<150x256x1x1xf32>, %arg204: tensor<150xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<1x3x512x512xf32>) -> tensor<1x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %378 = stablehlo.add %arg208, %377 : tensor<256xf32>
      %379 = stablehlo.rsqrt %378 : tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %379 = stablehlo.iota dim = 0 : tensor<128xi32>
      %380 = stablehlo.convert %379 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %382 = stablehlo.add %380, %381 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %386 = stablehlo.add %384, %385 : tensor<128xf32>
      %387 = stablehlo.maximum %378, %386 : tensor<128xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %389 = stablehlo.convert %388 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %390 = stablehlo.convert %389 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %391 = stablehlo.subtract %388, %390 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %393 = stablehlo.clamp %377, %391, %392 : tensor<128x1xf32>
      mhlo.return %393 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %379 = stablehlo.iota dim = 0 : tensor<128xi32>
      %380 = stablehlo.convert %379 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %382 = stablehlo.add %380, %381 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %386 = stablehlo.add %384, %385 : tensor<128xf32>
      %387 = stablehlo.maximum %378, %386 : tensor<128xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %389 = stablehlo.convert %388 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %390 = stablehlo.convert %389 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %391 = stablehlo.subtract %388, %390 : tensor<128x1xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %393 = stablehlo.clamp %377, %391, %392 : tensor<128x1xf32>
      mhlo.return %393 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %3 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %378 = stablehlo.iota dim = 0 : tensor<128xi32>
      %379 = stablehlo.convert %378 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %381 = stablehlo.add %379, %380 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %385 = stablehlo.add %383, %384 : tensor<128xf32>
      %386 = stablehlo.maximum %377, %385 : tensor<128xf32>
      %387 = stablehlo.convert %386 : (tensor<128xf32>) -> tensor<128xi32>
      %388 = stablehlo.convert %387 : (tensor<128xi32>) -> tensor<128xf32>
      %389 = stablehlo.subtract %386, %388 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %391 = stablehlo.clamp %377, %389, %390 : tensor<128xf32>
      mhlo.return %391 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %4 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg207, %arg0, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x512x512xf32>, tensor<32x3x7x7xf32>, tensor<32xf32>) -> tuple<tensor<1x32x128x128xf32>, tensor<0xui8>>
    %5 = stablehlo.get_tuple_element %4[0] : (tuple<tensor<1x32x128x128xf32>, tensor<0xui8>>) -> tensor<1x32x128x128xf32>
    %6 = "mhlo.fusion"(%5) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x128x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
      mhlo.return %378 : tensor<128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x128x128xf32>) -> tensor<128x128xf32>
    %7 = "mhlo.fusion"(%5, %6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x128x128xf32>, %arg209: tensor<128x128xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x128x128xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x32x128x128xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x32x128x128xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
      mhlo.return %386 : tensor<128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %8 = "mhlo.fusion"(%7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x128x128xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x128x128xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x128x128xf32>
      mhlo.return %382 : tensor<1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %9 = "mhlo.fusion"(%arg3, %arg2, %8, %5, %6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x128x128xf32>, %arg211: tensor<1x32x128x128xf32>, %arg212: tensor<128x128xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x128x128xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %383 = stablehlo.subtract %arg211, %382 : tensor<1x32x128x128xf32>
      %384 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %386 = mhlo.bitcast %385 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %387 = stablehlo.multiply %383, %386 : tensor<1x32x128x128xf32>
      %388 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %389 = stablehlo.multiply %387, %388 : tensor<1x32x128x128xf32>
      %390 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %391 = stablehlo.add %389, %390 : tensor<1x32x128x128xf32>
      %392 = mhlo.bitcast %391 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<32x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
      mhlo.return %393 : tensor<128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x128x128xf32>, tensor<1x32x128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %10 = "mhlo.fusion"(%arg3, %arg2, %8, %5, %9, %6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x128x128xf32>, %arg211: tensor<1x32x128x128xf32>, %arg212: tensor<128x128xf32>, %arg213: tensor<128x128xf32>):
      %377 = mhlo.bitcast %arg213 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x128x128xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %383 = stablehlo.subtract %arg211, %382 : tensor<1x32x128x128xf32>
      %384 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %386 = mhlo.bitcast %385 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %387 = stablehlo.multiply %383, %386 : tensor<1x32x128x128xf32>
      %388 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %389 = stablehlo.multiply %387, %388 : tensor<1x32x128x128xf32>
      %390 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %391 = stablehlo.add %389, %390 : tensor<1x32x128x128xf32>
      %392 = mhlo.bitcast %391 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %393 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %394 = stablehlo.multiply %393, %378 : tensor<1x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %397 = stablehlo.subtract %392, %396 : tensor<1x32x16384xf32>
      %398 = stablehlo.multiply %397, %397 : tensor<1x32x16384xf32>
      %399 = mhlo.bitcast %398 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %400 = stablehlo.reduce(%399 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %400 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x128x128xf32>, tensor<1x32x128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<16384xf32>
    %11:2 = "mhlo.fusion"(%arg17, %arg16, %arg3, %arg2, %8, %5, %9, %10, %6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32xf32>, %arg212: tensor<1x128x128xf32>, %arg213: tensor<1x32x128x128xf32>, %arg214: tensor<128x128xf32>, %arg215: tensor<16384xf32>, %arg216: tensor<128x128xf32>):
      %377 = mhlo.bitcast %arg216 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x128x128xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %383 = stablehlo.subtract %arg213, %382 : tensor<1x32x128x128xf32>
      %384 = mhlo.bitcast %arg212 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %386 = mhlo.bitcast %385 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %387 = stablehlo.multiply %383, %386 : tensor<1x32x128x128xf32>
      %388 = stablehlo.broadcast_in_dim %arg211, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %389 = stablehlo.multiply %387, %388 : tensor<1x32x128x128xf32>
      %390 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %391 = stablehlo.add %389, %390 : tensor<1x32x128x128xf32>
      %392 = mhlo.bitcast %391 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %393 = mhlo.bitcast %arg214 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %394 = stablehlo.multiply %393, %378 : tensor<1x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %397 = stablehlo.subtract %392, %396 : tensor<1x32x16384xf32>
      %398 = mhlo.bitcast %arg215 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %399 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384xf32>
      %403 = stablehlo.rsqrt %402 : tensor<1x16384xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %406 = stablehlo.multiply %397, %405 : tensor<1x32x16384xf32>
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<1x32x16384xf32>
      %409 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %410 = stablehlo.add %408, %409 : tensor<1x32x16384xf32>
      mhlo.return %410, %406 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x128x128xf32>, tensor<1x32x128x128xf32>, tensor<128x128xf32>, tensor<16384xf32>, tensor<128x128xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>)
    %12 = stablehlo.tuple %11#0, %11#1 {xla_shape = "(f32[1,32,16384]{2,1,0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<1x32x16384xf32>, tensor<1x32x16384xf32>>
    %13 = mhlo.bitcast %11#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<1x32x16384xf32>) -> tensor<16384x32xf32>
    %14 = "mhlo.fusion"(%13, %arg16, %arg17, %arg18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %381 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %15 = mhlo.bitcast %11#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %16 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%15, %arg24, %arg25) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<1x32x16x16xf32>, tensor<0xui8>>
    %17 = stablehlo.get_tuple_element %16[0] : (tuple<tensor<1x32x16x16xf32>, tensor<0xui8>>) -> tensor<1x32x16x16xf32>
    %18 = "mhlo.fusion"(%17) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16x16xf32>) -> tensor<16x16xf32>
    %19 = "mhlo.fusion"(%17, %18) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x32x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x32x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %20 = "mhlo.fusion"(%19, %17, %18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16xf32>, %arg209: tensor<1x32x16x16xf32>, %arg210: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %383 = stablehlo.subtract %arg209, %382 : tensor<1x32x16x16xf32>
      %384 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x16x16xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x16x16xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x32x16x16xf32>
      mhlo.return %392 : tensor<1x32x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>, tensor<1x32x16x16xf32>, tensor<16x16xf32>) -> tensor<1x32x16x16xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<1x32x16x16xf32>) -> tensor<256x32xf32>
    %22 = "mhlo.fusion"(%arg20, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %377 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %24 = "mhlo.fusion"(%21, %arg26, %arg27, %23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
      %381 = stablehlo.dot %380, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
      mhlo.return %381 : tensor<256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
    %25 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %377 = stablehlo.slice %arg208 [0:256, 0:32] : (tensor<256x64xf32>) -> tensor<256x32xf32>
      mhlo.return %377 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x32xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %27 = "mhlo.fusion"(%arg19, %14, %arg21, %26) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<32xf32>) -> tensor<32x256xf32>
      %380 = stablehlo.add %379, %arg211 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : tensor<32x256xf32>
      %381 = stablehlo.dot %378, %380, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<16384x256xf32>
      mhlo.return %383 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<16384x32xf32>, tensor<32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %28 = "mhlo.fusion"(%27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<16384x256xf32>
      %380 = stablehlo.exponential %379 : tensor<16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<16384x256xf32>
      mhlo.return %383 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>) -> tensor<16384x256xf32>
    %29 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %377 = stablehlo.slice %arg208 [0:256, 32:64] : (tensor<256x64xf32>) -> tensor<256x32xf32>
      mhlo.return %377 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x32xf32>
    %30 = "mhlo.fusion"(%28, %arg23, %29) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<256x32xf32>
      %379 = stablehlo.dot %arg208, %378, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %379 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    %31 = "mhlo.fusion"(%30, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %377 : tensor<16384x32xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %32:2 = "mhlo.fusion"(%arg3, %arg2, %8, %5, %6, %31, %arg29) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x128x128xf32>, %arg211: tensor<1x32x128x128xf32>, %arg212: tensor<128x128xf32>, %arg213: tensor<16384x32xf32>, %arg214: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg214, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<1x128x128xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %386 = mhlo.bitcast %385 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %387 = stablehlo.subtract %arg211, %386 : tensor<1x32x128x128xf32>
      %388 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x128xf32>) -> tensor<16384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x32x128x128xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x32x128x128xf32>
      %394 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x128x128xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x32x128x128xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %397 = stablehlo.add %380, %396 : tensor<1x32x16384xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %399, %397 : tensor<16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x128x128xf32>, tensor<1x32x128x128xf32>, tensor<128x128xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>)
    %33 = stablehlo.tuple %32#0, %32#1 {xla_shape = "(f32[16384]{0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<16384xf32>, tensor<1x32x16384xf32>>
    %34 = "mhlo.fusion"(%32#1, %32#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x32x16384xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x32x16384xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %385 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384xf32>) -> tensor<16384xf32>
    %35 = "mhlo.fusion"(%32#1, %34, %32#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384xf32>, %arg210: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x32x16384xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x16384xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x16384xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x16384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x32x16384xf32>
      mhlo.return %390 : tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384xf32>, tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<1x32x16384xf32>) -> tensor<16384x32xf32>
    %37 = "mhlo.fusion"(%36, %arg30, %arg31, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
      mhlo.return %381 : tensor<16384x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    %38 = "mhlo.fusion"(%37, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<128xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<128x16384xf32>
      mhlo.return %379 : tensor<128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<128xf32>) -> tensor<128x16384xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x16384xf32>) -> tensor<1x128x128x128xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%39, %arg34) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>) -> tensor<1x128x128x128xf32>
    %42 = "mhlo.fusion"(%41, %arg35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x128x128x128xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x128x128x128xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %382 = stablehlo.negate %379 : tensor<1x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x128x128x128xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %386 = stablehlo.abs %385 : tensor<1x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x128x16384xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x128x16384xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x128x16384xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x128x16384xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x128x16384xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x128x16384xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x128x16384xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x128x16384xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %414 = stablehlo.negate %390 : tensor<1x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %417 = stablehlo.exponential %414 : tensor<1x128x16384xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x128x16384xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x128x16384xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x128x16384xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x128x16384xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x128x16384xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x128x16384xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x128x16384xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x128x16384xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x128x16384xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x128x16384xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x128x16384xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x128x16384xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x128x16384xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x128x16384xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x128x16384xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x128x16384xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x128x16384xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x128x16384xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x128x16384xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x128x16384xf32>
      mhlo.return %475 : tensor<1x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x128xf32>, tensor<128xf32>) -> tensor<1x128x16384xf32>
    %43 = mhlo.bitcast %42 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<1x128x16384xf32>) -> tensor<16384x128xf32>
    %44 = "mhlo.fusion"(%43, %arg36) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<32x128xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
      mhlo.return %377 : tensor<16384x32xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    %45:2 = "mhlo.fusion"(%32#1, %44, %arg37) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %383, %381 : tensor<16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>)
    %46 = stablehlo.tuple %45#0, %45#1 {xla_shape = "(f32[16384]{0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<16384xf32>, tensor<1x32x16384xf32>>
    %47 = "mhlo.fusion"(%45#1, %45#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x32x16384xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x32x16384xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %385 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384xf32>) -> tensor<16384xf32>
    %48:2 = "mhlo.fusion"(%arg39, %arg38, %45#1, %45#0, %47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x32x16384xf32>, %arg211: tensor<16384xf32>, %arg212: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg210, %381 : tensor<1x32x16384xf32>
      %383 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x16384xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x16384xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x16384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x32x16384xf32>
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %392 = stablehlo.multiply %390, %391 : tensor<1x32x16384xf32>
      %393 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %394 = stablehlo.add %392, %393 : tensor<1x32x16384xf32>
      mhlo.return %394, %390 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x32x16384xf32>, tensor<16384xf32>, tensor<16384xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>)
    %49 = stablehlo.tuple %48#0, %48#1 {xla_shape = "(f32[1,32,16384]{2,1,0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<1x32x16384xf32>, tensor<1x32x16384xf32>>
    %50 = mhlo.bitcast %48#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<1x32x16384xf32>) -> tensor<16384x32xf32>
    %51 = "mhlo.fusion"(%50, %arg38, %arg39, %arg40) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %381 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %52 = mhlo.bitcast %48#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %53 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%52, %arg46, %arg47) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,32,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x32x128x128xf32>, tensor<32x32x8x8xf32>, tensor<32xf32>) -> tuple<tensor<1x32x16x16xf32>, tensor<0xui8>>
    %54 = stablehlo.get_tuple_element %53[0] : (tuple<tensor<1x32x16x16xf32>, tensor<0xui8>>) -> tensor<1x32x16x16xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16x16xf32>) -> tensor<16x16xf32>
    %56 = "mhlo.fusion"(%54, %55) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x32x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x32x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<32x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %57 = "mhlo.fusion"(%56, %54, %55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16xf32>, %arg209: tensor<1x32x16x16xf32>, %arg210: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %383 = stablehlo.subtract %arg209, %382 : tensor<1x32x16x16xf32>
      %384 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x16x16xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x16x16xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<256xf32>) -> tensor<1x32x256xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x256xf32>) -> tensor<1x32x16x16xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x32x16x16xf32>
      mhlo.return %392 : tensor<1x32x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>, tensor<1x32x16x16xf32>, tensor<16x16xf32>) -> tensor<1x32x16x16xf32>
    %58 = mhlo.bitcast %57 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<1x32x16x16xf32>) -> tensor<256x32xf32>
    %59 = "mhlo.fusion"(%arg42, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
      mhlo.return %377 : tensor<64x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %60 = mhlo.bitcast %59 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,64]{0,1}"} : (tensor<64x32xf32>) -> tensor<32x64xf32>
    %61 = "mhlo.fusion"(%58, %arg48, %arg49, %60) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : (tensor<32xf32>) -> tensor<256x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,32]{0,1}"} : tensor<256x32xf32>
      %381 = stablehlo.dot %380, %arg211, precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
      mhlo.return %381 : tensor<256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x64xf32>) -> tensor<256x64xf32>
    %62 = "mhlo.fusion"(%61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %377 = stablehlo.slice %arg208 [0:256, 0:32] : (tensor<256x64xf32>) -> tensor<256x32xf32>
      mhlo.return %377 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x32xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %64 = "mhlo.fusion"(%arg41, %51, %arg43, %63) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<32xf32>) -> tensor<32x256xf32>
      %380 = stablehlo.add %379, %arg211 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : tensor<32x256xf32>
      %381 = stablehlo.dot %378, %380, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<16384x256xf32>
      mhlo.return %383 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<16384x32xf32>, tensor<32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %65 = "mhlo.fusion"(%64) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<16384x256xf32>
      %380 = stablehlo.exponential %379 : tensor<16384x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<16384x256xf32>, tensor<f32>) -> tensor<16384xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0] : (tensor<16384xf32>) -> tensor<16384x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<16384x256xf32>
      mhlo.return %383 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>) -> tensor<16384x256xf32>
    %66 = "mhlo.fusion"(%61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %377 = stablehlo.slice %arg208 [0:256, 32:64] : (tensor<256x64xf32>) -> tensor<256x32xf32>
      mhlo.return %377 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x32xf32>
    %67 = "mhlo.fusion"(%65, %arg45, %66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<32xf32>, %arg210: tensor<256x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<256x32xf32>
      %379 = stablehlo.dot %arg208, %378, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %379 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    %68 = "mhlo.fusion"(%67, %arg50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %377 : tensor<16384x32xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %69:2 = "mhlo.fusion"(%45#1, %68, %arg51) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %383, %381 : tensor<16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> (tensor<16384xf32>, tensor<1x32x16384xf32>)
    %70 = stablehlo.tuple %69#0, %69#1 {xla_shape = "(f32[16384]{0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<16384xf32>, tensor<1x32x16384xf32>>
    %71 = "mhlo.fusion"(%69#1, %69#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x32x16384xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x32x16384xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %385 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384xf32>) -> tensor<16384xf32>
    %72 = "mhlo.fusion"(%69#1, %71, %69#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384xf32>, %arg210: tensor<16384xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x32x16384xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x16384xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x16384xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x16384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x32x16384xf32>
      mhlo.return %390 : tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384xf32>, tensor<16384xf32>) -> tensor<1x32x16384xf32>
    %73 = mhlo.bitcast %72 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<1x32x16384xf32>) -> tensor<16384x32xf32>
    %74 = "mhlo.fusion"(%73, %arg52, %arg53, %arg54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<128x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
      mhlo.return %381 : tensor<16384x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    %75 = "mhlo.fusion"(%74, %arg55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<128xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<128x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<128x16384xf32>
      mhlo.return %379 : tensor<128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<128xf32>) -> tensor<128x16384xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x16384xf32>) -> tensor<1x128x128x128xf32>
    %77 = stablehlo.custom_call @__cudnn$convForward(%76, %arg56) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>
    %78 = stablehlo.get_tuple_element %77[0] : (tuple<tensor<1x128x128x128xf32>, tensor<0xui8>>) -> tensor<1x128x128x128xf32>
    %79 = "mhlo.fusion"(%78, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x128x128x128xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x128x128x128xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %382 = stablehlo.negate %379 : tensor<1x128x128x128xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x128x128x128xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %386 = stablehlo.abs %385 : tensor<1x128x16384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x128x16384xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x128x16384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x128x16384xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x128x16384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x128x16384xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x128x16384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x128x16384xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x128x16384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x128x16384xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x128x16384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x128x16384xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x128x16384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x128x16384xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x128x16384xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x128x16384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %414 = stablehlo.negate %390 : tensor<1x128x16384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %417 = stablehlo.exponential %414 : tensor<1x128x16384xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x128x16384xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x128x16384xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x128x16384xf32>, tensor<1x128x16384xf32>) -> tensor<1x128x16384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x128x16384xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x128x16384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x128x16384xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x128x16384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x128x16384xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x128x16384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x128x16384xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x128x16384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x128x16384xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x128x16384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x128x16384xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x128x16384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x128x16384xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x128x16384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x128x16384xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x128x16384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x128x16384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x128x16384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x128x16384xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x128x16384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x128x16384xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x128x16384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x128x16384xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x128x16384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x128x16384xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x128x16384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x128x16384xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x128x16384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x128x16384xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x128x16384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x128x16384xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x128x16384xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x128x16384xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x128x16384xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x128x16384xi1>, tensor<1x128x16384xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x128x16384xf32>
      mhlo.return %475 : tensor<1x128x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x128xf32>, tensor<128xf32>) -> tensor<1x128x16384xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,128]{0,1}"} : (tensor<1x128x16384xf32>) -> tensor<16384x128xf32>
    %81 = "mhlo.fusion"(%80, %arg58) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<32x128xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
      mhlo.return %377 : tensor<16384x32xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    %82 = "mhlo.fusion"(%69#1, %81, %arg59) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x32x16384xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %383 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x16384xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> tensor<16384xf32>
    %83 = "mhlo.fusion"(%82, %69#1, %81, %arg59) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<16384xf32>, %arg209: tensor<1x32x16384xf32>, %arg210: tensor<16384x32xf32>, %arg211: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg211, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = stablehlo.add %380, %arg209 : tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x16384xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %387 = stablehlo.subtract %381, %386 : tensor<1x32x16384xf32>
      %388 = stablehlo.multiply %387, %387 : tensor<1x32x16384xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<32x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %390 = stablehlo.reduce(%389 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32x16384xf32>, tensor<f32>) -> tensor<16384xf32>
      mhlo.return %390 : tensor<16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16384xf32>, tensor<1x32x16384xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> tensor<16384xf32>
    %84:2 = "mhlo.fusion"(%arg185, %arg184, %82, %83, %69#1, %81, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16384xf32>, %arg211: tensor<16384xf32>, %arg212: tensor<1x32x16384xf32>, %arg213: tensor<16384x32xf32>, %arg214: tensor<32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg214, dims = [0] : (tensor<32xf32>) -> tensor<32x16384xf32>
      %378 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<32x16384xf32>
      %379 = stablehlo.add %377, %378 : tensor<32x16384xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x16384xf32>) -> tensor<1x32x16384xf32>
      %381 = stablehlo.add %380, %arg212 : tensor<1x32x16384xf32>
      %382 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x16384xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %387 = stablehlo.subtract %381, %386 : tensor<1x32x16384xf32>
      %388 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<16384xf32>) -> tensor<1x16384xf32>
      %389 = stablehlo.multiply %388, %383 : tensor<1x16384xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %391 = stablehlo.add %389, %390 : tensor<1x16384xf32>
      %392 = stablehlo.rsqrt %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [2] : (tensor<16384xf32>) -> tensor<1x32x16384xf32>
      %395 = stablehlo.multiply %387, %394 : tensor<1x32x16384xf32>
      %396 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x32x16384xf32>
      %398 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<32xf32>) -> tensor<1x32x16384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x32x16384xf32>
      mhlo.return %399, %395 : tensor<1x32x16384xf32>, tensor<1x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16384xf32>, tensor<16384xf32>, tensor<1x32x16384xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> (tensor<1x32x16384xf32>, tensor<1x32x16384xf32>)
    %85 = stablehlo.tuple %84#0, %84#1 {xla_shape = "(f32[1,32,16384]{2,1,0}, f32[1,32,16384]{2,1,0})"} : tuple<tensor<1x32x16384xf32>, tensor<1x32x16384xf32>>
    %86 = mhlo.bitcast %84#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
    %87 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%86, %arg4, %arg5) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x32x128x128xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tuple<tensor<1x64x64x64xf32>, tensor<0xui8>>
    %88 = stablehlo.get_tuple_element %87[0] : (tuple<tensor<1x64x64x64xf32>, tensor<0xui8>>) -> tensor<1x64x64x64xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x64x64xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
      mhlo.return %378 : tensor<64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x64xf32>) -> tensor<64x64xf32>
    %90 = "mhlo.fusion"(%88, %89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x64x64xf32>, %arg209: tensor<64x64xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x64x64xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x64x64x64xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x64x64x64xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
      mhlo.return %386 : tensor<64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %91:2 = "mhlo.fusion"(%arg7, %arg6, %90, %88, %89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x64xf32>, %arg211: tensor<1x64x64x64xf32>, %arg212: tensor<64x64xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x64x64xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
      %383 = stablehlo.subtract %arg211, %382 : tensor<1x64x64x64xf32>
      %384 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x64x64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x64x64xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x64x64xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %394 = stablehlo.multiply %392, %393 : tensor<1x64x64x64xf32>
      %395 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x64x64x64xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<64x64x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %398 = stablehlo.reduce(%397 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<64x64x64xf32>, tensor<f32>) -> tensor<64x64xf32>
      mhlo.return %398, %396 : tensor<64x64xf32>, tensor<1x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<1x64x64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<1x64x64x64xf32>)
    %92 = stablehlo.tuple %91#0, %91#1 {xla_shape = "(f32[64,64]{1,0}, f32[1,64,64,64]{3,2,1,0})"} : tuple<tensor<64x64xf32>, tensor<1x64x64x64xf32>>
    %93 = mhlo.bitcast %arg63 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %94 = "mhlo.fusion"(%91#1, %91#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x64x64xf32>, %arg209: tensor<64x64xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x64x64xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %383 = stablehlo.subtract %377, %382 : tensor<1x64x4096xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x64x4096xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %386 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x64xf32>, tensor<64x64xf32>) -> tensor<4096xf32>
    %95:2 = "mhlo.fusion"(%arg61, %arg60, %91#1, %94, %91#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x64x64x64xf32>, %arg211: tensor<4096xf32>, %arg212: tensor<64x64xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %378 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x64x64xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x64x64xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x64xf32>) -> tensor<4096xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %383 = stablehlo.subtract %377, %382 : tensor<1x64x4096xf32>
      %384 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %385 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %386 = stablehlo.multiply %384, %385 : tensor<1x4096xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %388 = stablehlo.add %386, %387 : tensor<1x4096xf32>
      %389 = stablehlo.rsqrt %388 : tensor<1x4096xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x64x4096xf32>
      %393 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %394 = stablehlo.multiply %392, %393 : tensor<1x64x4096xf32>
      %395 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x64x4096xf32>
      mhlo.return %396, %392 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x64x64xf32>, tensor<4096xf32>, tensor<64x64xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>)
    %96 = stablehlo.tuple %95#0, %95#1 {xla_shape = "(f32[1,64,4096]{2,1,0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<1x64x4096xf32>, tensor<1x64x4096xf32>>
    %97 = mhlo.bitcast %95#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<1x64x4096xf32>) -> tensor<4096x64xf32>
    %98 = "mhlo.fusion"(%97, %arg60, %arg61, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %381 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %99 = mhlo.bitcast %98 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<4096x2x32xf32>
    %100 = mhlo.bitcast %arg65 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %101 = mhlo.bitcast %95#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %102 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%101, %arg68, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<1x64x16x16xf32>, tensor<0xui8>>
    %103 = stablehlo.get_tuple_element %102[0] : (tuple<tensor<1x64x16x16xf32>, tensor<0xui8>>) -> tensor<1x64x16x16xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x16x16xf32>) -> tensor<16x16xf32>
    %105 = "mhlo.fusion"(%103, %104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x64x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x64x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %106 = "mhlo.fusion"(%105, %103, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16xf32>, %arg209: tensor<1x64x16x16xf32>, %arg210: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %383 = stablehlo.subtract %arg209, %382 : tensor<1x64x16x16xf32>
      %384 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x16x16xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x16x16xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x64x16x16xf32>
      mhlo.return %392 : tensor<1x64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>, tensor<1x64x16x16xf32>, tensor<16x16xf32>) -> tensor<1x64x16x16xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<1x64x16x16xf32>) -> tensor<256x64xf32>
    %108 = "mhlo.fusion"(%107, %arg70, %arg71, %arg64, %arg66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
      %381 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %383 = stablehlo.dot %380, %382, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64x128xf32>) -> tensor<256x128xf32>
      mhlo.return %383 : tensor<256x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<256x128xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
      %378 = stablehlo.slice %377 [0:64, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
      mhlo.return %378 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<64x256xf32>
    %110 = mhlo.bitcast %109 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<2x32x256xf32>
    %111 = "mhlo.fusion"(%93, %99, %100, %110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x32xf32>, %arg209: tensor<4096x2x32xf32>, %arg210: tensor<2x32xf32>, %arg211: tensor<2x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2] : (tensor<2x32xf32>) -> tensor<4096x2x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<4096x2x32xf32>
      %379 = stablehlo.transpose %378, dims = [1, 0, 2] : (tensor<4096x2x32xf32>) -> tensor<2x4096x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<2x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<2x4096x256xf32>
      mhlo.return %384 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x32xf32>, tensor<4096x2x32xf32>, tensor<2x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %112 = "mhlo.fusion"(%111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<2x4096x256xf32>
      %380 = stablehlo.exponential %379 : tensor<2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<2x4096x256xf32>
      mhlo.return %383 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>) -> tensor<2x4096x256xf32>
    %113 = mhlo.bitcast %arg67 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %114 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
      %378 = stablehlo.slice %377 [64:128, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
      mhlo.return %378 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<64x256xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<2x32x256xf32>
    %116 = "mhlo.fusion"(%112, %113, %115) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>, %arg209: tensor<2x32xf32>, %arg210: tensor<2x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<2x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<2x32x256xf32>) -> tensor<2x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<2x4096x32xf32>) -> tensor<4096x2x32xf32>
      mhlo.return %381 : tensor<4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>, tensor<2x32xf32>, tensor<2x32x256xf32>) -> tensor<4096x2x32xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4096x2x32xf32>) -> tensor<4096x64xf32>
    %118 = "mhlo.fusion"(%117, %arg72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64x64xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %377 : tensor<4096x64xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %119:2 = "mhlo.fusion"(%91#1, %118, %arg73) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x64x64xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.add %380, %381 : tensor<1x64x4096xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %384, %382 : tensor<4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x64xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>)
    %120 = stablehlo.tuple %119#0, %119#1 {xla_shape = "(f32[4096]{0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<4096xf32>, tensor<1x64x4096xf32>>
    %121 = "mhlo.fusion"(%119#1, %119#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x64x4096xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x64x4096xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %385 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %122 = "mhlo.fusion"(%119#1, %121, %119#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096xf32>, %arg210: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x64x4096xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x4096xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x4096xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x4096xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x64x4096xf32>
      mhlo.return %390 : tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %123 = mhlo.bitcast %122 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<1x64x4096xf32>) -> tensor<4096x64xf32>
    %124 = "mhlo.fusion"(%123, %arg74, %arg75, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
      mhlo.return %381 : tensor<4096x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %125 = "mhlo.fusion"(%124, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<256xf32>) -> tensor<256x4096xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x4096xf32>
      mhlo.return %379 : tensor<256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<256x4096xf32>
    %126 = mhlo.bitcast %125 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<1x256x64x64xf32>
    %127 = stablehlo.custom_call @__cudnn$convForward(%126, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>
    %128 = stablehlo.get_tuple_element %127[0] : (tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>) -> tensor<1x256x64x64xf32>
    %129 = "mhlo.fusion"(%128, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x256x64x64xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x256x64x64xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %382 = stablehlo.negate %379 : tensor<1x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x256x64x64xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %386 = stablehlo.abs %385 : tensor<1x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x256x4096xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x256x4096xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x256x4096xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x256x4096xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x4096xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x256x4096xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x256x4096xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x256x4096xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %414 = stablehlo.negate %390 : tensor<1x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %417 = stablehlo.exponential %414 : tensor<1x256x4096xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x256x4096xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x256x4096xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x256x4096xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x256x4096xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x256x4096xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x256x4096xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x256x4096xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x256x4096xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x256x4096xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x256x4096xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x256x4096xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x256x4096xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x256x4096xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x256x4096xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x256x4096xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x256x4096xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x256x4096xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x256x4096xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x256x4096xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x256x4096xf32>
      mhlo.return %475 : tensor<1x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64x64xf32>, tensor<256xf32>) -> tensor<1x256x4096xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<1x256x4096xf32>) -> tensor<4096x256xf32>
    %131 = "mhlo.fusion"(%130, %arg80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<64x256xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
      mhlo.return %377 : tensor<4096x64xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    %132:2 = "mhlo.fusion"(%119#1, %131, %arg81) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %383, %381 : tensor<4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>)
    %133 = stablehlo.tuple %132#0, %132#1 {xla_shape = "(f32[4096]{0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<4096xf32>, tensor<1x64x4096xf32>>
    %134 = mhlo.bitcast %arg85 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %135 = "mhlo.fusion"(%132#1, %132#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x64x4096xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x64x4096xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %385 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %136:2 = "mhlo.fusion"(%arg83, %arg82, %132#1, %135, %132#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x64x4096xf32>, %arg211: tensor<4096xf32>, %arg212: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg210, %381 : tensor<1x64x4096xf32>
      %383 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x4096xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x4096xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x4096xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x64x4096xf32>
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %392 = stablehlo.multiply %390, %391 : tensor<1x64x4096xf32>
      %393 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %394 = stablehlo.add %392, %393 : tensor<1x64x4096xf32>
      mhlo.return %394, %390 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x64x4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>)
    %137 = stablehlo.tuple %136#0, %136#1 {xla_shape = "(f32[1,64,4096]{2,1,0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<1x64x4096xf32>, tensor<1x64x4096xf32>>
    %138 = mhlo.bitcast %136#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<1x64x4096xf32>) -> tensor<4096x64xf32>
    %139 = "mhlo.fusion"(%138, %arg82, %arg83, %arg84) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %381 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %140 = mhlo.bitcast %139 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<4096x2x32xf32>
    %141 = mhlo.bitcast %arg87 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %142 = mhlo.bitcast %136#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %143 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%142, %arg90, %arg91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,64,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x64x64xf32>, tensor<64x64x4x4xf32>, tensor<64xf32>) -> tuple<tensor<1x64x16x16xf32>, tensor<0xui8>>
    %144 = stablehlo.get_tuple_element %143[0] : (tuple<tensor<1x64x16x16xf32>, tensor<0xui8>>) -> tensor<1x64x16x16xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x16x16xf32>) -> tensor<16x16xf32>
    %146 = "mhlo.fusion"(%144, %145) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x64x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x64x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<64x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %147 = "mhlo.fusion"(%146, %144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16xf32>, %arg209: tensor<1x64x16x16xf32>, %arg210: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %383 = stablehlo.subtract %arg209, %382 : tensor<1x64x16x16xf32>
      %384 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x16x16xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x16x16xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<256xf32>) -> tensor<1x64x256xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x256xf32>) -> tensor<1x64x16x16xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x64x16x16xf32>
      mhlo.return %392 : tensor<1x64x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16xf32>, tensor<1x64x16x16xf32>, tensor<16x16xf32>) -> tensor<1x64x16x16xf32>
    %148 = mhlo.bitcast %147 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<1x64x16x16xf32>) -> tensor<256x64xf32>
    %149 = "mhlo.fusion"(%148, %arg92, %arg93, %arg86, %arg88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x64xf32>, %arg212: tensor<64x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : (tensor<64xf32>) -> tensor<256x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,64]{0,1}"} : tensor<256x64xf32>
      %381 = stablehlo.concatenate %arg211, %arg212, dim = 0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,128]{0,1}"} : (tensor<128x64xf32>) -> tensor<64x128xf32>
      %383 = stablehlo.dot %380, %382, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64x128xf32>) -> tensor<256x128xf32>
      mhlo.return %383 : tensor<256x128xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,128]{0,1}"} : (tensor<256x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<256x128xf32>
    %150 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
      %378 = stablehlo.slice %377 [0:64, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
      mhlo.return %378 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<64x256xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<2x32x256xf32>
    %152 = "mhlo.fusion"(%134, %140, %141, %151) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x32xf32>, %arg209: tensor<4096x2x32xf32>, %arg210: tensor<2x32xf32>, %arg211: tensor<2x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2] : (tensor<2x32xf32>) -> tensor<4096x2x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<4096x2x32xf32>
      %379 = stablehlo.transpose %378, dims = [1, 0, 2] : (tensor<4096x2x32xf32>) -> tensor<2x4096x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<2x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<2x4096x256xf32>
      mhlo.return %384 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x32xf32>, tensor<4096x2x32xf32>, tensor<2x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %153 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<2x4096x256xf32>
      %380 = stablehlo.exponential %379 : tensor<2x4096x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x4096x256xf32>, tensor<f32>) -> tensor<2x4096xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<2x4096xf32>) -> tensor<2x4096x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<2x4096x256xf32>
      mhlo.return %383 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>) -> tensor<2x4096x256xf32>
    %154 = mhlo.bitcast %arg89 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<64xf32>) -> tensor<2x32xf32>
    %155 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x128xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
      %378 = stablehlo.slice %377 [64:128, 0:256] : (tensor<128x256xf32>) -> tensor<64x256xf32>
      mhlo.return %378 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<64x256xf32>
    %156 = mhlo.bitcast %155 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<2x32x256xf32>
    %157 = "mhlo.fusion"(%153, %154, %156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>, %arg209: tensor<2x32xf32>, %arg210: tensor<2x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<2x32xf32>) -> tensor<2x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<2x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<2x32x256xf32>) -> tensor<2x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<2x4096x32xf32>) -> tensor<4096x2x32xf32>
      mhlo.return %381 : tensor<4096x2x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>, tensor<2x32xf32>, tensor<2x32x256xf32>) -> tensor<4096x2x32xf32>
    %158 = mhlo.bitcast %157 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4096x2x32xf32>) -> tensor<4096x64xf32>
    %159 = "mhlo.fusion"(%158, %arg94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64x64xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %377 : tensor<4096x64xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %160:2 = "mhlo.fusion"(%132#1, %159, %arg95) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %383, %381 : tensor<4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> (tensor<4096xf32>, tensor<1x64x4096xf32>)
    %161 = stablehlo.tuple %160#0, %160#1 {xla_shape = "(f32[4096]{0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<4096xf32>, tensor<1x64x4096xf32>>
    %162 = "mhlo.fusion"(%160#1, %160#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x64x4096xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x64x4096xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %385 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    %163 = "mhlo.fusion"(%160#1, %162, %160#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096xf32>, %arg210: tensor<4096xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x64x4096xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x4096xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x4096xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x4096xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x64x4096xf32>
      mhlo.return %390 : tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<1x64x4096xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<1x64x4096xf32>) -> tensor<4096x64xf32>
    %165 = "mhlo.fusion"(%164, %arg96, %arg97, %arg98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
      mhlo.return %381 : tensor<4096x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %166 = "mhlo.fusion"(%165, %arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<256xf32>) -> tensor<256x4096xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<256x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x4096xf32>
      mhlo.return %379 : tensor<256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<256x4096xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x4096xf32>) -> tensor<1x256x64x64xf32>
    %168 = stablehlo.custom_call @__cudnn$convForward(%167, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>
    %169 = stablehlo.get_tuple_element %168[0] : (tuple<tensor<1x256x64x64xf32>, tensor<0xui8>>) -> tensor<1x256x64x64xf32>
    %170 = "mhlo.fusion"(%169, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64x64xf32>, %arg209: tensor<256xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64x64xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x256x64x64xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x256x64x64xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %382 = stablehlo.negate %379 : tensor<1x256x64x64xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x256x64x64xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %386 = stablehlo.abs %385 : tensor<1x256x4096xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x256x4096xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x256x4096xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x256x4096xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x256x4096xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x256x4096xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x256x4096xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x256x4096xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x256x4096xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x4096xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x256x4096xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x256x4096xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x256x4096xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x256x4096xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x256x4096xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x256x4096xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %414 = stablehlo.negate %390 : tensor<1x256x4096xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %417 = stablehlo.exponential %414 : tensor<1x256x4096xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x256x4096xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x256x4096xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x256x4096xf32>, tensor<1x256x4096xf32>) -> tensor<1x256x4096xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x256x4096xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x256x4096xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x256x4096xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x256x4096xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x256x4096xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x256x4096xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x256x4096xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x256x4096xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x256x4096xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x256x4096xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x256x4096xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x256x4096xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x256x4096xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x256x4096xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x256x4096xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x256x4096xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x256x4096xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x256x4096xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x256x4096xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x256x4096xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x256x4096xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x256x4096xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x256x4096xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x256x4096xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x256x4096xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x256x4096xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x256x4096xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x256x4096xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x256x4096xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x256x4096xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x4096xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x256x4096xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x256x4096xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x256x4096xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x256x4096xi1>, tensor<1x256x4096xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x256x4096xf32>
      mhlo.return %475 : tensor<1x256x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64x64xf32>, tensor<256xf32>) -> tensor<1x256x4096xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,256]{0,1}"} : (tensor<1x256x4096xf32>) -> tensor<4096x256xf32>
    %172 = "mhlo.fusion"(%171, %arg102) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<64x256xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
      mhlo.return %377 : tensor<4096x64xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    %173 = "mhlo.fusion"(%160#1, %172, %arg103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x64x4096xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %383 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x4096xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> tensor<4096xf32>
    %174 = "mhlo.fusion"(%173, %160#1, %172, %arg103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<4096xf32>, %arg209: tensor<1x64x4096xf32>, %arg210: tensor<4096x64xf32>, %arg211: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg211, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = stablehlo.add %380, %arg209 : tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x4096xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %387 = stablehlo.subtract %381, %386 : tensor<1x64x4096xf32>
      %388 = stablehlo.multiply %387, %387 : tensor<1x64x4096xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<64x4096xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %390 = stablehlo.reduce(%389 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<64x4096xf32>, tensor<f32>) -> tensor<4096xf32>
      mhlo.return %390 : tensor<4096xf32>
    }) {output_operand_aliasing = []} : (tensor<4096xf32>, tensor<1x64x4096xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> tensor<4096xf32>
    %175:2 = "mhlo.fusion"(%arg187, %arg186, %174, %173, %160#1, %172, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<4096xf32>, %arg211: tensor<4096xf32>, %arg212: tensor<1x64x4096xf32>, %arg213: tensor<4096x64xf32>, %arg214: tensor<64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg214, dims = [0] : (tensor<64xf32>) -> tensor<64x4096xf32>
      %378 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<64x4096xf32>
      %379 = stablehlo.add %377, %378 : tensor<64x4096xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x4096xf32>) -> tensor<1x64x4096xf32>
      %381 = stablehlo.add %380, %arg212 : tensor<1x64x4096xf32>
      %382 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x4096xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %387 = stablehlo.subtract %381, %386 : tensor<1x64x4096xf32>
      %388 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<4096xf32>) -> tensor<1x4096xf32>
      %389 = stablehlo.multiply %388, %383 : tensor<1x4096xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %391 = stablehlo.add %389, %390 : tensor<1x4096xf32>
      %392 = stablehlo.rsqrt %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [2] : (tensor<4096xf32>) -> tensor<1x64x4096xf32>
      %395 = stablehlo.multiply %387, %394 : tensor<1x64x4096xf32>
      %396 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x64x4096xf32>
      %398 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<64xf32>) -> tensor<1x64x4096xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x64x4096xf32>
      mhlo.return %399, %395 : tensor<1x64x4096xf32>, tensor<1x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<4096xf32>, tensor<4096xf32>, tensor<1x64x4096xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> (tensor<1x64x4096xf32>, tensor<1x64x4096xf32>)
    %176 = stablehlo.tuple %175#0, %175#1 {xla_shape = "(f32[1,64,4096]{2,1,0}, f32[1,64,4096]{2,1,0})"} : tuple<tensor<1x64x4096xf32>, tensor<1x64x4096xf32>>
    %177 = mhlo.bitcast %175#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
    %178 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%177, %arg8, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,160,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x64x64x64xf32>, tensor<160x64x3x3xf32>, tensor<160xf32>) -> tuple<tensor<1x160x32x32xf32>, tensor<0xui8>>
    %179 = stablehlo.get_tuple_element %178[0] : (tuple<tensor<1x160x32x32xf32>, tensor<0xui8>>) -> tensor<1x160x32x32xf32>
    %180 = "mhlo.fusion"(%179) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x32x32xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
      mhlo.return %378 : tensor<32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x32x32xf32>) -> tensor<32x32xf32>
    %181 = "mhlo.fusion"(%179, %180) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x32x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x32x32xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x160x32x32xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x160x32x32xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
      mhlo.return %386 : tensor<32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %182 = "mhlo.fusion"(%arg11, %arg10, %179, %181, %180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x160x32x32xf32>, %arg211: tensor<32x32xf32>, %arg212: tensor<32x32xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x32x32xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
      %383 = stablehlo.subtract %arg210, %382 : tensor<1x160x32x32xf32>
      %384 = mhlo.bitcast %arg211 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
      %385 = stablehlo.multiply %384, %378 : tensor<1x32x32xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
      %387 = stablehlo.add %385, %386 : tensor<1x32x32xf32>
      %388 = stablehlo.rsqrt %387 : tensor<1x32x32xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %391 = mhlo.bitcast %390 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x160x32x32xf32>
      %393 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1x160x32x32xf32>
      %394 = stablehlo.multiply %392, %393 : tensor<1x160x32x32xf32>
      %395 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<160xf32>) -> tensor<1x160x32x32xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x160x32x32xf32>
      mhlo.return %396 : tensor<1x160x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x160x32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<1x160x32x32xf32>
    %183 = mhlo.bitcast %arg107 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %184 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x32x32xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<160x32x32xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x32x32xf32>, tensor<f32>) -> tensor<32x32xf32>
      mhlo.return %378 : tensor<32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x32x32xf32>) -> tensor<32x32xf32>
    %185 = "mhlo.fusion"(%182, %184) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x32x32xf32>, %arg209: tensor<32x32xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x32x32xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %383 = stablehlo.subtract %377, %382 : tensor<1x160x1024xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x160x1024xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %386 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x32x32xf32>, tensor<32x32xf32>) -> tensor<1024xf32>
    %186:2 = "mhlo.fusion"(%arg105, %arg104, %182, %185, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x160x32x32xf32>, %arg211: tensor<1024xf32>, %arg212: tensor<32x32xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %378 = mhlo.bitcast %arg212 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x32x32xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x32xf32>) -> tensor<1024xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %383 = stablehlo.subtract %377, %382 : tensor<1x160x1024xf32>
      %384 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %385 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %386 = stablehlo.multiply %384, %385 : tensor<1x1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %388 = stablehlo.add %386, %387 : tensor<1x1024xf32>
      %389 = stablehlo.rsqrt %388 : tensor<1x1024xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %392 = stablehlo.multiply %383, %391 : tensor<1x160x1024xf32>
      %393 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %394 = stablehlo.multiply %392, %393 : tensor<1x160x1024xf32>
      %395 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x160x1024xf32>
      mhlo.return %396, %392 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x160x32x32xf32>, tensor<1024xf32>, tensor<32x32xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>)
    %187 = stablehlo.tuple %186#0, %186#1 {xla_shape = "(f32[1,160,1024]{2,1,0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1x160x1024xf32>, tensor<1x160x1024xf32>>
    %188 = mhlo.bitcast %186#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1x160x1024xf32>) -> tensor<1024x160xf32>
    %189 = "mhlo.fusion"(%188, %arg104, %arg105, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %381 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %190 = mhlo.bitcast %189 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1024x5x32xf32>
    %191 = mhlo.bitcast %arg109 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %192 = mhlo.bitcast %186#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %193 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%192, %arg112, %arg113) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<1x160x16x16xf32>, tensor<0xui8>>
    %194 = stablehlo.get_tuple_element %193[0] : (tuple<tensor<1x160x16x16xf32>, tensor<0xui8>>) -> tensor<1x160x16x16xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>) -> tensor<16x16xf32>
    %196:2 = "mhlo.fusion"(%194, %195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x160x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x160x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386, %383 : tensor<16x16xf32>, tensor<1x160x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>, tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<1x160x16x16xf32>)
    %197 = stablehlo.tuple %196#0, %196#1 {xla_shape = "(f32[16,16]{1,0}, f32[1,160,16,16]{3,2,1,0})"} : tuple<tensor<16x16xf32>, tensor<1x160x16x16xf32>>
    %198 = "mhlo.fusion"(%196#1, %196#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x16x16xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x16x16xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
      %386 = stablehlo.multiply %arg208, %385 : tensor<1x160x16x16xf32>
      mhlo.return %386 : tensor<1x160x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>, tensor<16x16xf32>) -> tensor<1x160x16x16xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<1x160x16x16xf32>) -> tensor<256x160xf32>
    %200 = "mhlo.fusion"(%arg108, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %377 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %377 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %202 = "mhlo.fusion"(%199, %arg114, %arg115, %201) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
      %381 = stablehlo.dot %380, %arg211, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
      mhlo.return %381 : tensor<256x320xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
    %203 = "mhlo.fusion"(%202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x320xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
      %378 = stablehlo.slice %377 [0:160, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
      mhlo.return %378 : tensor<160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x320xf32>) -> tensor<160x256xf32>
    %204 = mhlo.bitcast %203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x256xf32>) -> tensor<5x32x256xf32>
    %205 = "mhlo.fusion"(%183, %190, %191, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x32xf32>, %arg209: tensor<1024x5x32xf32>, %arg210: tensor<5x32xf32>, %arg211: tensor<5x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2] : (tensor<5x32xf32>) -> tensor<1024x5x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<1024x5x32xf32>
      %379 = stablehlo.transpose %378, dims = [1, 0, 2] : (tensor<1024x5x32xf32>) -> tensor<5x1024x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<5x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<5x1024x256xf32>
      mhlo.return %384 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x32xf32>, tensor<1024x5x32xf32>, tensor<5x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %206 = "mhlo.fusion"(%205) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<5x1024x256xf32>
      %380 = stablehlo.exponential %379 : tensor<5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<5x1024x256xf32>
      mhlo.return %383 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>) -> tensor<5x1024x256xf32>
    %207 = mhlo.bitcast %arg111 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %208 = "mhlo.fusion"(%202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x320xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
      %378 = stablehlo.slice %377 [160:320, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
      mhlo.return %378 : tensor<160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x320xf32>) -> tensor<160x256xf32>
    %209 = mhlo.bitcast %208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x256xf32>) -> tensor<5x32x256xf32>
    %210 = "mhlo.fusion"(%206, %207, %209) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>, %arg209: tensor<5x32xf32>, %arg210: tensor<5x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<5x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<5x32x256xf32>) -> tensor<5x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<5x1024x32xf32>) -> tensor<1024x5x32xf32>
      mhlo.return %381 : tensor<1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>, tensor<5x32xf32>, tensor<5x32x256xf32>) -> tensor<1024x5x32xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x5x32xf32>) -> tensor<1024x160xf32>
    %212 = "mhlo.fusion"(%211, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160x160xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %377 : tensor<1024x160xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %213:2 = "mhlo.fusion"(%182, %212, %arg117) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x32x32xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<160x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
      %381 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.add %380, %381 : tensor<1x160x1024xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %384, %382 : tensor<1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x32x32xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>)
    %214 = stablehlo.tuple %213#0, %213#1 {xla_shape = "(f32[1024]{0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1024xf32>, tensor<1x160x1024xf32>>
    %215 = "mhlo.fusion"(%213#1, %213#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x160x1024xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %385 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %216 = "mhlo.fusion"(%213#1, %215, %213#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x1024xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x1024xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x160x1024xf32>
      mhlo.return %390 : tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1x160x1024xf32>) -> tensor<1024x160xf32>
    %218 = "mhlo.fusion"(%217, %arg118, %arg119, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
      mhlo.return %381 : tensor<1024x640xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    %219 = "mhlo.fusion"(%218, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<640xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<640xf32>) -> tensor<640x1024xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<640x1024xf32>
      mhlo.return %379 : tensor<640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x640xf32>, tensor<640xf32>) -> tensor<640x1024xf32>
    %220 = mhlo.bitcast %219 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<1x640x32x32xf32>
    %221 = stablehlo.custom_call @__cudnn$convForward(%220, %arg122) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<1x640x32x32xf32>, tensor<0xui8>>
    %222 = stablehlo.get_tuple_element %221[0] : (tuple<tensor<1x640x32x32xf32>, tensor<0xui8>>) -> tensor<1x640x32x32xf32>
    %223 = "mhlo.fusion"(%222, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<1x640x32x32xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x640x32x32xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x640x32x32xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %382 = stablehlo.negate %379 : tensor<1x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x640x32x32xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %386 = stablehlo.abs %385 : tensor<1x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x640x1024xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x640x1024xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x640x1024xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x640x1024xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x640x1024xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x640x1024xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x640x1024xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x640x1024xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %414 = stablehlo.negate %390 : tensor<1x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %417 = stablehlo.exponential %414 : tensor<1x640x1024xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x640x1024xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x640x1024xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x640x1024xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x640x1024xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x640x1024xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x640x1024xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x640x1024xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x640x1024xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x640x1024xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x640x1024xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x640x1024xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x640x1024xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x640x1024xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x640x1024xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x640x1024xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x640x1024xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x640x1024xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x640x1024xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x640x1024xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x640x1024xf32>
      mhlo.return %475 : tensor<1x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x640x32x32xf32>, tensor<640xf32>) -> tensor<1x640x1024xf32>
    %224 = mhlo.bitcast %223 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1x640x1024xf32>) -> tensor<1024x640xf32>
    %225 = "mhlo.fusion"(%224, %arg124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<160x640xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
      mhlo.return %377 : tensor<1024x160xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    %226:2 = "mhlo.fusion"(%213#1, %225, %arg125) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<160x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x160x1024xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %383, %381 : tensor<1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>)
    %227 = stablehlo.tuple %226#0, %226#1 {xla_shape = "(f32[1024]{0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1024xf32>, tensor<1x160x1024xf32>>
    %228 = mhlo.bitcast %arg129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %229 = "mhlo.fusion"(%226#1, %226#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x160x1024xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %385 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %230:2 = "mhlo.fusion"(%arg127, %arg126, %226#1, %229, %226#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x160x1024xf32>, %arg211: tensor<1024xf32>, %arg212: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg210, %381 : tensor<1x160x1024xf32>
      %383 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x1024xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x1024xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x160x1024xf32>
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %392 = stablehlo.multiply %390, %391 : tensor<1x160x1024xf32>
      %393 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %394 = stablehlo.add %392, %393 : tensor<1x160x1024xf32>
      mhlo.return %394, %390 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x160x1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>)
    %231 = stablehlo.tuple %230#0, %230#1 {xla_shape = "(f32[1,160,1024]{2,1,0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1x160x1024xf32>, tensor<1x160x1024xf32>>
    %232 = mhlo.bitcast %230#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1x160x1024xf32>) -> tensor<1024x160xf32>
    %233 = "mhlo.fusion"(%232, %arg126, %arg127, %arg128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %381 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %234 = mhlo.bitcast %233 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1024x5x32xf32>
    %235 = mhlo.bitcast %arg131 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %236 = mhlo.bitcast %230#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %237 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%236, %arg134, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,160,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x160x32x32xf32>, tensor<160x160x2x2xf32>, tensor<160xf32>) -> tuple<tensor<1x160x16x16xf32>, tensor<0xui8>>
    %238 = stablehlo.get_tuple_element %237[0] : (tuple<tensor<1x160x16x16xf32>, tensor<0xui8>>) -> tensor<1x160x16x16xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>) -> tensor<16x16xf32>
    %240:2 = "mhlo.fusion"(%238, %239) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x160x16x16xf32>
      %384 = stablehlo.multiply %383, %383 : tensor<1x160x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<160x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%385 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %386, %383 : tensor<16x16xf32>, tensor<1x160x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>, tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<1x160x16x16xf32>)
    %241 = stablehlo.tuple %240#0, %240#1 {xla_shape = "(f32[16,16]{1,0}, f32[1,160,16,16]{3,2,1,0})"} : tuple<tensor<16x16xf32>, tensor<1x160x16x16xf32>>
    %242 = "mhlo.fusion"(%240#1, %240#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x160x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x16x16xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x16x16xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x160x256xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x256xf32>) -> tensor<1x160x16x16xf32>
      %386 = stablehlo.multiply %arg208, %385 : tensor<1x160x16x16xf32>
      mhlo.return %386 : tensor<1x160x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x16x16xf32>, tensor<16x16xf32>) -> tensor<1x160x16x16xf32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<1x160x16x16xf32>) -> tensor<256x160xf32>
    %244 = "mhlo.fusion"(%arg130, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160xf32>, %arg209: tensor<160x160xf32>):
      %377 = stablehlo.concatenate %arg208, %arg209, dim = 0 : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
      mhlo.return %377 : tensor<320x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160xf32>, tensor<160x160xf32>) -> tensor<320x160xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[160,320]{0,1}"} : (tensor<320x160xf32>) -> tensor<160x320xf32>
    %246 = "mhlo.fusion"(%243, %arg136, %arg137, %245) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<160x320xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : (tensor<160xf32>) -> tensor<256x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,160]{0,1}"} : tensor<256x160xf32>
      %381 = stablehlo.dot %380, %arg211, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
      mhlo.return %381 : tensor<256x320xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,320]{0,1}"} : (tensor<256x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160x320xf32>) -> tensor<256x320xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x320xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
      %378 = stablehlo.slice %377 [0:160, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
      mhlo.return %378 : tensor<160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x320xf32>) -> tensor<160x256xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x256xf32>) -> tensor<5x32x256xf32>
    %249 = "mhlo.fusion"(%228, %234, %235, %248) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x32xf32>, %arg209: tensor<1024x5x32xf32>, %arg210: tensor<5x32xf32>, %arg211: tensor<5x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [1, 2] : (tensor<5x32xf32>) -> tensor<1024x5x32xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<1024x5x32xf32>
      %379 = stablehlo.transpose %378, dims = [1, 0, 2] : (tensor<1024x5x32xf32>) -> tensor<5x1024x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<5x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<5x1024x256xf32>
      mhlo.return %384 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x32xf32>, tensor<1024x5x32xf32>, tensor<5x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<5x1024x256xf32>
      %380 = stablehlo.exponential %379 : tensor<5x1024x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<5x1024x256xf32>, tensor<f32>) -> tensor<5x1024xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<5x1024xf32>) -> tensor<5x1024x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<5x1024x256xf32>
      mhlo.return %383 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>) -> tensor<5x1024x256xf32>
    %251 = mhlo.bitcast %arg133 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<160xf32>) -> tensor<5x32xf32>
    %252 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x320xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x320xf32>) -> tensor<320x256xf32>
      %378 = stablehlo.slice %377 [160:320, 0:256] : (tensor<320x256xf32>) -> tensor<160x256xf32>
      mhlo.return %378 : tensor<160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x320xf32>) -> tensor<160x256xf32>
    %253 = mhlo.bitcast %252 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x256xf32>) -> tensor<5x32x256xf32>
    %254 = "mhlo.fusion"(%250, %251, %253) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>, %arg209: tensor<5x32xf32>, %arg210: tensor<5x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<5x32xf32>) -> tensor<5x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<5x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<5x32x256xf32>) -> tensor<5x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<5x1024x32xf32>) -> tensor<1024x5x32xf32>
      mhlo.return %381 : tensor<1024x5x32xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>, tensor<5x32xf32>, tensor<5x32x256xf32>) -> tensor<1024x5x32xf32>
    %255 = mhlo.bitcast %254 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x5x32xf32>) -> tensor<1024x160xf32>
    %256 = "mhlo.fusion"(%255, %arg138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160x160xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %377 : tensor<1024x160xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %257:2 = "mhlo.fusion"(%226#1, %256, %arg139) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<160x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x160x1024xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %383, %381 : tensor<1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>)
    %258 = stablehlo.tuple %257#0, %257#1 {xla_shape = "(f32[1024]{0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1024xf32>, tensor<1x160x1024xf32>>
    %259 = "mhlo.fusion"(%257#1, %257#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x160x1024xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %385 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %260 = "mhlo.fusion"(%257#1, %259, %257#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x1024xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x1024xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x160x1024xf32>
      mhlo.return %390 : tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x160x1024xf32>
    %261 = mhlo.bitcast %260 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1x160x1024xf32>) -> tensor<1024x160xf32>
    %262 = "mhlo.fusion"(%261, %arg140, %arg141, %arg142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<640x160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
      mhlo.return %381 : tensor<1024x640xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    %263 = "mhlo.fusion"(%262, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<640xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<640xf32>) -> tensor<640x1024xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<640x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<640x1024xf32>
      mhlo.return %379 : tensor<640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x640xf32>, tensor<640xf32>) -> tensor<640x1024xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<640x1024xf32>) -> tensor<1x640x32x32xf32>
    %265 = stablehlo.custom_call @__cudnn$convForward(%264, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,640,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1x640x32x32xf32>, tensor<640x1x3x3xf32>) -> tuple<tensor<1x640x32x32xf32>, tensor<0xui8>>
    %266 = stablehlo.get_tuple_element %265[0] : (tuple<tensor<1x640x32x32xf32>, tensor<0xui8>>) -> tensor<1x640x32x32xf32>
    %267 = "mhlo.fusion"(%266, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x640x32x32xf32>, %arg209: tensor<640xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<1x640x32x32xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x640x32x32xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x640x32x32xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %382 = stablehlo.negate %379 : tensor<1x640x32x32xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x640x32x32xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x640x32x32xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %386 = stablehlo.abs %385 : tensor<1x640x1024xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x640x1024xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x640x1024xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x640x1024xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x640x1024xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x640x1024xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x640x1024xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x640x1024xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x640x1024xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x640x1024xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x640x1024xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x640x1024xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x640x1024xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x640x1024xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x640x1024xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x640x1024xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %414 = stablehlo.negate %390 : tensor<1x640x1024xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %417 = stablehlo.exponential %414 : tensor<1x640x1024xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x640x1024xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x640x1024xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x640x1024xf32>, tensor<1x640x1024xf32>) -> tensor<1x640x1024xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x640x1024xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x640x1024xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x640x1024xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x640x1024xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x640x1024xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x640x1024xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x640x1024xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x640x1024xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x640x1024xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x640x1024xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x640x1024xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x640x1024xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x640x1024xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x640x1024xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x640x1024xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x640x1024xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x640x1024xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x640x1024xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x640x1024xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x640x1024xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x640x1024xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x640x1024xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x640x1024xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x640x1024xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x640x1024xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x640x1024xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x640x1024xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x640x1024xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x640x1024xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x640x1024xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x640x1024xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x640x1024xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x640x1024xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x640x1024xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x640x1024xi1>, tensor<1x640x1024xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x640x1024xf32>
      mhlo.return %475 : tensor<1x640x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x640x32x32xf32>, tensor<640xf32>) -> tensor<1x640x1024xf32>
    %268 = mhlo.bitcast %267 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,640]{0,1}"} : (tensor<1x640x1024xf32>) -> tensor<1024x640xf32>
    %269 = "mhlo.fusion"(%268, %arg146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<160x640xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
      mhlo.return %377 : tensor<1024x160xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    %270:2 = "mhlo.fusion"(%257#1, %269, %arg147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<160xf32>) -> tensor<160x1024xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<160x1024xf32>
      %379 = stablehlo.add %377, %378 : tensor<160x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<160x1024xf32>) -> tensor<1x160x1024xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x160x1024xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %383, %381 : tensor<1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> (tensor<1024xf32>, tensor<1x160x1024xf32>)
    %271 = stablehlo.tuple %270#0, %270#1 {xla_shape = "(f32[1024]{0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1024xf32>, tensor<1x160x1024xf32>>
    %272 = "mhlo.fusion"(%270#1, %270#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x160x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x160x1024xf32>
      %383 = stablehlo.multiply %382, %382 : tensor<1x160x1024xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<160x1024xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.reduce(%384 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<160x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %385 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x160x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %273:2 = "mhlo.fusion"(%arg189, %arg188, %270#1, %272, %270#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x160x1024xf32>, %arg211: tensor<1024xf32>, %arg212: tensor<1024xf32>):
      %377 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x1024xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %382 = stablehlo.subtract %arg210, %381 : tensor<1x160x1024xf32>
      %383 = mhlo.bitcast %arg211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1024xf32>) -> tensor<1x1024xf32>
      %384 = stablehlo.multiply %383, %378 : tensor<1x1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x1024xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x1024xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<1024xf32>) -> tensor<1x160x1024xf32>
      %390 = stablehlo.multiply %382, %389 : tensor<1x160x1024xf32>
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %392 = stablehlo.multiply %390, %391 : tensor<1x160x1024xf32>
      %393 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<160xf32>) -> tensor<1x160x1024xf32>
      %394 = stablehlo.add %392, %393 : tensor<1x160x1024xf32>
      mhlo.return %394, %390 : tensor<1x160x1024xf32>, tensor<1x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x160x1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1x160x1024xf32>, tensor<1x160x1024xf32>)
    %274 = stablehlo.tuple %273#0, %273#1 {xla_shape = "(f32[1,160,1024]{2,1,0}, f32[1,160,1024]{2,1,0})"} : tuple<tensor<1x160x1024xf32>, tensor<1x160x1024xf32>>
    %275 = mhlo.bitcast %273#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<1x160x1024xf32>) -> tensor<1024x160xf32>
    %276 = "mhlo.fusion"(%275, %arg188, %arg189, %arg196) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<256x160xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : (tensor<160xf32>) -> tensor<1024x160xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[1024,160]{0,1}"} : tensor<1024x160xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<256x160xf32>) -> tensor<1024x256xf32>
      mhlo.return %381 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<256x160xf32>) -> tensor<1024x256xf32>
    %277 = "mhlo.fusion"(%276, %arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x256xf32>, %arg209: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1024x256xf32>
      %378 = stablehlo.add %arg208, %377 : tensor<1024x256xf32>
      mhlo.return %378 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>, tensor<256xf32>) -> tensor<1024x256xf32>
    %278 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %379 = stablehlo.iota dim = 0 : tensor<128xi32>
      %380 = stablehlo.convert %379 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %382 = stablehlo.add %380, %381 : tensor<128xf32>
      %383 = stablehlo.multiply %382, %381 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %385 = stablehlo.add %383, %384 : tensor<128xf32>
      %386 = stablehlo.maximum %378, %385 : tensor<128xf32>
      %387 = mhlo.bitcast %386 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %388 = stablehlo.convert %387 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %389 = stablehlo.convert %388 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %390 = stablehlo.subtract %387, %389 : tensor<128x1xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %392 = stablehlo.clamp %377, %390, %391 : tensor<128x1xf32>
      mhlo.return %392 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %279 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %379 = stablehlo.iota dim = 0 : tensor<128xi32>
      %380 = stablehlo.convert %379 : (tensor<128xi32>) -> tensor<128xf32>
      %381 = stablehlo.maximum %378, %380 : tensor<128xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %383 = stablehlo.convert %382 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %384 = stablehlo.convert %383 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %385 = stablehlo.subtract %382, %384 : tensor<128x1xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %387 = stablehlo.clamp %377, %385, %386 : tensor<128x1xf32>
      mhlo.return %387 : tensor<128x1xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128x1xf32>
    %280 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %378 = stablehlo.iota dim = 0 : tensor<128xi32>
      %379 = stablehlo.convert %378 : (tensor<128xi32>) -> tensor<128xf32>
      %380 = stablehlo.maximum %377, %379 : tensor<128xf32>
      %381 = stablehlo.convert %380 : (tensor<128xf32>) -> tensor<128xi32>
      %382 = stablehlo.convert %381 : (tensor<128xi32>) -> tensor<128xf32>
      %383 = stablehlo.subtract %380, %382 : tensor<128xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %385 = stablehlo.clamp %377, %383, %384 : tensor<128xf32>
      mhlo.return %385 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %281 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %378 = stablehlo.iota dim = 0 : tensor<128xi32>
      %379 = stablehlo.convert %378 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %381 = stablehlo.add %379, %380 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %385 = stablehlo.add %383, %384 : tensor<128xf32>
      %386 = stablehlo.maximum %377, %385 : tensor<128xf32>
      %387 = stablehlo.convert %386 : (tensor<128xf32>) -> tensor<128xi32>
      %388 = stablehlo.convert %387 : (tensor<128xi32>) -> tensor<128xf32>
      %389 = stablehlo.subtract %386, %388 : tensor<128xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %390 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %391 = stablehlo.clamp %377, %389, %390 : tensor<128xf32>
      mhlo.return %391 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %282 = mhlo.bitcast %273#0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
    %283 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%282, %arg12, %arg13) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x160x32x32xf32>, tensor<256x160x3x3xf32>, tensor<256xf32>) -> tuple<tensor<1x256x16x16xf32>, tensor<0xui8>>
    %284 = stablehlo.get_tuple_element %283[0] : (tuple<tensor<1x256x16x16xf32>, tensor<0xui8>>) -> tensor<1x256x16x16xf32>
    %285 = "mhlo.fusion"(%284) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>) -> tensor<16x16xf32>
    %286 = "mhlo.fusion"(%284, %285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x256x16x16xf32>
      %383 = stablehlo.subtract %arg208, %382 : tensor<1x256x16x16xf32>
      mhlo.return %383 : tensor<1x256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>, tensor<16x16xf32>) -> tensor<1x256x16x16xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x16x16xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %379 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>) -> tensor<16x16xf32>
    %288 = "mhlo.fusion"(%286, %arg15, %arg14, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg211 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x16x16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x16x16xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x16x16xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x256x16x16xf32>
      %386 = stablehlo.multiply %arg208, %385 : tensor<1x256x16x16xf32>
      %387 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<1x256x16x16xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x256x16x16xf32>
      %389 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1x256x16x16xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x256x16x16xf32>
      mhlo.return %390 : tensor<1x256x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x16xf32>) -> tensor<1x256x16x16xf32>
    %289 = mhlo.bitcast %arg151 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %290 = "mhlo.fusion"(%288) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16x16xf32>, tensor<f32>) -> tensor<16x16xf32>
      mhlo.return %378 : tensor<16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>) -> tensor<16x16xf32>
    %291 = "mhlo.fusion"(%288, %290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>, %arg209: tensor<16x16xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x16x16xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<256xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %383 = stablehlo.subtract %377, %382 : tensor<1x256x256xf32>
      mhlo.return %383 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>, tensor<16x16xf32>) -> tensor<1x256x256xf32>
    %292 = "mhlo.fusion"(%291) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x256xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %293 = "mhlo.fusion"(%291, %292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x256xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x256xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = stablehlo.multiply %arg208, %384 : tensor<1x256x256xf32>
      mhlo.return %385 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %295 = "mhlo.fusion"(%294, %arg148, %arg149, %arg150, %arg152, %arg154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>, %arg212: tensor<256x256xf32>, %arg213: tensor<256x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %381 = stablehlo.concatenate %arg211, %arg212, %arg213, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
      %383 = stablehlo.dot %380, %382, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256x768xf32>) -> tensor<256x768xf32>
      mhlo.return %383 : tensor<256x768xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x768xf32>
    %296 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [0:256, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %298 = mhlo.bitcast %arg153 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %299 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [256:512, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %301 = "mhlo.fusion"(%289, %297, %298, %300) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x32xf32>, %arg209: tensor<8x32x256xf32>, %arg210: tensor<8x32xf32>, %arg211: tensor<8x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<8x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<8x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<8x256x256xf32>
      mhlo.return %384 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x32xf32>, tensor<8x32x256xf32>, tensor<8x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %302 = "mhlo.fusion"(%301) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<8x256x256xf32>
      %380 = stablehlo.exponential %379 : tensor<8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<8x256x256xf32>
      mhlo.return %383 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
    %303 = mhlo.bitcast %arg155 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %304 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [512:768, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %305 = mhlo.bitcast %304 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %306 = "mhlo.fusion"(%302, %303, %305) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x32xf32>, %arg210: tensor<8x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<8x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<8x256x32xf32>) -> tensor<256x8x32xf32>
      mhlo.return %381 : tensor<256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x32xf32>, tensor<8x32x256xf32>) -> tensor<256x8x32xf32>
    %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x32xf32>) -> tensor<256x256xf32>
    %308 = "mhlo.fusion"(%307, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x256xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %377 : tensor<256x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %309 = "mhlo.fusion"(%288, %308, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x16x16xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %381 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %382 = stablehlo.add %380, %381 : tensor<1x256x256xf32>
      mhlo.return %382 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x16x16xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %310 = "mhlo.fusion"(%309) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %378 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %311 = "mhlo.fusion"(%309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x256x256xf32>
      mhlo.return %382 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %312 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x256xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %313 = "mhlo.fusion"(%311, %312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x256xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x256xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = stablehlo.multiply %arg208, %384 : tensor<1x256x256xf32>
      mhlo.return %385 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %314 = mhlo.bitcast %313 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %315 = "mhlo.fusion"(%314, %arg158, %arg159, %arg160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
      mhlo.return %381 : tensor<256x1024xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %316 = "mhlo.fusion"(%315, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<1024x256xf32>
      mhlo.return %379 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %317 = mhlo.bitcast %316 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x1024x16x16xf32>
    %318 = stablehlo.custom_call @__cudnn$convForward(%317, %arg162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<1x1024x16x16xf32>, tensor<0xui8>>
    %319 = stablehlo.get_tuple_element %318[0] : (tuple<tensor<1x1024x16x16xf32>, tensor<0xui8>>) -> tensor<1x1024x16x16xf32>
    %320 = "mhlo.fusion"(%319, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x16x16xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x1024x16x16xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x1024x16x16xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %382 = stablehlo.negate %379 : tensor<1x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x1024x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %386 = stablehlo.abs %385 : tensor<1x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x1024x256xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x1024x256xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x1024x256xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x1024x256xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x256xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x1024x256xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x1024x256xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x1024x256xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %414 = stablehlo.negate %390 : tensor<1x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %417 = stablehlo.exponential %414 : tensor<1x1024x256xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x1024x256xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x1024x256xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x1024x256xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x1024x256xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x1024x256xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x1024x256xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x1024x256xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x1024x256xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x1024x256xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x1024x256xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x1024x256xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x1024x256xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x1024x256xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x1024x256xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x1024x256xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x1024x256xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x1024x256xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x1024x256xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x1024x256xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x1024x256xf32>
      mhlo.return %475 : tensor<1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x16x16xf32>, tensor<1024xf32>) -> tensor<1x1024x256xf32>
    %321 = mhlo.bitcast %320 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<1x1024x256xf32>) -> tensor<256x1024xf32>
    %322 = "mhlo.fusion"(%321, %arg164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<256x1024xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
      mhlo.return %377 : tensor<256x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    %323 = "mhlo.fusion"(%309, %322, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x256x256xf32>
      mhlo.return %381 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %324 = mhlo.bitcast %arg169 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %325 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %378 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %326 = "mhlo.fusion"(%323, %325) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x256x256xf32>
      mhlo.return %382 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %327 = "mhlo.fusion"(%326) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x256xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %328 = "mhlo.fusion"(%326, %327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x256xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x256xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = stablehlo.multiply %arg208, %384 : tensor<1x256x256xf32>
      mhlo.return %385 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %329 = mhlo.bitcast %328 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %330 = "mhlo.fusion"(%329, %arg166, %arg167, %arg168, %arg170, %arg172) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>, %arg212: tensor<256x256xf32>, %arg213: tensor<256x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %381 = stablehlo.concatenate %arg211, %arg212, %arg213, dim = 0 : (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<768x256xf32>
      %382 = mhlo.bitcast %381 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<768x256xf32>) -> tensor<256x768xf32>
      %383 = stablehlo.dot %380, %382, precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256x768xf32>) -> tensor<256x768xf32>
      mhlo.return %383 : tensor<256x768xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,768]{0,1}"} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x768xf32>
    %331 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [0:256, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %332 = mhlo.bitcast %331 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %333 = mhlo.bitcast %arg171 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %334 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [256:512, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %335 = mhlo.bitcast %334 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %336 = "mhlo.fusion"(%324, %332, %333, %335) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x32xf32>, %arg209: tensor<8x32x256xf32>, %arg210: tensor<8x32xf32>, %arg211: tensor<8x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg208, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %378 = stablehlo.add %377, %arg209 : tensor<8x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %380 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %381 = stablehlo.add %380, %arg211 : tensor<8x32x256xf32>
      %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
      %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<8x256x256xf32>
      mhlo.return %384 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x32xf32>, tensor<8x32x256xf32>, tensor<8x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %337 = "mhlo.fusion"(%336) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %377 = stablehlo.reduce(%arg208 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %379 = stablehlo.subtract %arg208, %378 : tensor<8x256x256xf32>
      %380 = stablehlo.exponential %379 : tensor<8x256x256xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.reduce(%380 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %383 = stablehlo.divide %380, %382 : tensor<8x256x256xf32>
      mhlo.return %383 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
    %338 = mhlo.bitcast %arg173 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<8x32xf32>
    %339 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x768xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<768x256xf32>
      %378 = stablehlo.slice %377 [512:768, 0:256] : (tensor<768x256xf32>) -> tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x768xf32>) -> tensor<256x256xf32>
    %340 = mhlo.bitcast %339 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<8x32x256xf32>
    %341 = "mhlo.fusion"(%337, %338, %340) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x32xf32>, %arg210: tensor<8x32x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1] : (tensor<8x32xf32>) -> tensor<8x32x256xf32>
      %378 = stablehlo.add %377, %arg210 : tensor<8x32x256xf32>
      %379 = stablehlo.transpose %378, dims = [0, 2, 1] : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %380 = stablehlo.dot_general %arg208, %379, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
      %381 = stablehlo.transpose %380, dims = [1, 0, 2] : (tensor<8x256x32xf32>) -> tensor<256x8x32xf32>
      mhlo.return %381 : tensor<256x8x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x32xf32>, tensor<8x32x256xf32>) -> tensor<256x8x32xf32>
    %342 = mhlo.bitcast %341 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x32xf32>) -> tensor<256x256xf32>
    %343 = "mhlo.fusion"(%342, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256x256xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %377 : tensor<256x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %344 = "mhlo.fusion"(%323, %343, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x256x256xf32>
      mhlo.return %381 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %345 = "mhlo.fusion"(%344) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %378 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %346 = "mhlo.fusion"(%344, %345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x256x256xf32>
      mhlo.return %382 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x256xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %348 = "mhlo.fusion"(%346, %347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x256xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x256xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = stablehlo.multiply %arg208, %384 : tensor<1x256x256xf32>
      mhlo.return %385 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %349 = mhlo.bitcast %348 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %350 = "mhlo.fusion"(%349, %arg176, %arg177, %arg178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1024x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
      mhlo.return %381 : tensor<256x1024xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %351 = "mhlo.fusion"(%350, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<1024xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256xf32>
      %378 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<1024x256xf32>
      mhlo.return %379 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %352 = mhlo.bitcast %351 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x1024x16x16xf32>
    %353 = stablehlo.custom_call @__cudnn$convForward(%352, %arg180) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,1024,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x16x16xf32>, tensor<1024x1x3x3xf32>) -> tuple<tensor<1x1024x16x16xf32>, tensor<0xui8>>
    %354 = stablehlo.get_tuple_element %353[0] : (tuple<tensor<1x1024x16x16xf32>, tensor<0xui8>>) -> tensor<1x1024x16x16xf32>
    %355 = "mhlo.fusion"(%354, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x16x16xf32>, %arg209: tensor<1024xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
      %378 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x16x16xf32>
      %379 = stablehlo.add %arg208, %378 : tensor<1x1024x16x16xf32>
      %380 = stablehlo.multiply %377, %379 : tensor<1x1024x16x16xf32>
      %381 = mhlo.bitcast %380 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %382 = stablehlo.negate %379 : tensor<1x1024x16x16xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1024x16x16xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x1024x16x16xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %386 = stablehlo.abs %385 : tensor<1x1024x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %388 = stablehlo.compare LT, %386, %387 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %390 = stablehlo.multiply %385, %385 : tensor<1x1024x256xf32>
      %391 = stablehlo.multiply %389, %390 : tensor<1x1024x256xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x1024x256xf32>
      %394 = stablehlo.multiply %393, %390 : tensor<1x1024x256xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %396 = stablehlo.add %394, %395 : tensor<1x1024x256xf32>
      %397 = stablehlo.multiply %396, %390 : tensor<1x1024x256xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x1024x256xf32>
      %400 = stablehlo.multiply %399, %390 : tensor<1x1024x256xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x256xf32>
      %403 = stablehlo.multiply %402, %390 : tensor<1x1024x256xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x1024x256xf32>
      %406 = stablehlo.multiply %405, %390 : tensor<1x1024x256xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x1024x256xf32>
      %409 = stablehlo.multiply %385, %408 : tensor<1x1024x256xf32>
      %410 = stablehlo.subtract %387, %409 : tensor<1x1024x256xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %412 = stablehlo.compare LT, %385, %411 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %414 = stablehlo.negate %390 : tensor<1x1024x256xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %417 = stablehlo.exponential %414 : tensor<1x1024x256xf32>
      %418 = stablehlo.divide %387, %386 : tensor<1x1024x256xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x1024x256xf32>
      %420 = stablehlo.compare LT, %386, %413 : (tensor<1x1024x256xf32>, tensor<1x1024x256xf32>) -> tensor<1x1024x256xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %422 = stablehlo.divide %387, %390 : tensor<1x1024x256xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<1x1024x256xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %425 = stablehlo.add %423, %424 : tensor<1x1024x256xf32>
      %426 = stablehlo.multiply %425, %422 : tensor<1x1024x256xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %428 = stablehlo.add %426, %427 : tensor<1x1024x256xf32>
      %429 = stablehlo.multiply %428, %422 : tensor<1x1024x256xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x1024x256xf32>
      %432 = stablehlo.multiply %431, %422 : tensor<1x1024x256xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x1024x256xf32>
      %435 = stablehlo.multiply %434, %422 : tensor<1x1024x256xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x1024x256xf32>
      %438 = stablehlo.multiply %437, %422 : tensor<1x1024x256xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x1024x256xf32>
      %441 = stablehlo.multiply %440, %422 : tensor<1x1024x256xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x1024x256xf32>
      %444 = stablehlo.multiply %443, %422 : tensor<1x1024x256xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x1024x256xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %448 = stablehlo.multiply %447, %422 : tensor<1x1024x256xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x1024x256xf32>
      %451 = stablehlo.multiply %450, %422 : tensor<1x1024x256xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x1024x256xf32>
      %454 = stablehlo.multiply %453, %422 : tensor<1x1024x256xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x1024x256xf32>
      %457 = stablehlo.multiply %456, %422 : tensor<1x1024x256xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x1024x256xf32>
      %460 = stablehlo.multiply %459, %422 : tensor<1x1024x256xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x1024x256xf32>
      %463 = stablehlo.multiply %462, %422 : tensor<1x1024x256xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x1024x256xf32>
      %466 = stablehlo.multiply %465, %422 : tensor<1x1024x256xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x256xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x1024x256xf32>
      %469 = stablehlo.select %420, %446, %468 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %470 = stablehlo.multiply %419, %469 : tensor<1x1024x256xf32>
      %471 = stablehlo.select %416, %411, %470 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %472 = stablehlo.subtract %413, %471 : tensor<1x1024x256xf32>
      %473 = stablehlo.select %412, %472, %471 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %474 = stablehlo.select %388, %410, %473 : tensor<1x1024x256xi1>, tensor<1x1024x256xf32>
      %475 = stablehlo.multiply %381, %474 : tensor<1x1024x256xf32>
      mhlo.return %475 : tensor<1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x16x16xf32>, tensor<1024xf32>) -> tensor<1x1024x256xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,1024]{0,1}"} : (tensor<1x1024x256xf32>) -> tensor<256x1024xf32>
    %357 = "mhlo.fusion"(%356, %arg182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<256x1024xf32>):
      %377 = stablehlo.dot_general %arg208, %arg209, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
      mhlo.return %377 : tensor<256x256xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    %358 = "mhlo.fusion"(%344, %357, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg210, dims = [0] : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %379 = stablehlo.add %377, %378 : tensor<256x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %381 = stablehlo.add %380, %arg208 : tensor<1x256x256xf32>
      mhlo.return %381 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %359 = "mhlo.fusion"(%358) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.reduce(%377 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %378 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %360 = "mhlo.fusion"(%358, %359) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %380 = mhlo.bitcast %379 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %382 = stablehlo.subtract %arg208, %381 : tensor<1x256x256xf32>
      mhlo.return %382 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %361 = "mhlo.fusion"(%360) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>):
      %377 = stablehlo.multiply %arg208, %arg208 : tensor<1x256x256xf32>
      %378 = mhlo.bitcast %377 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %379 = stablehlo.reduce(%378 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x256xf32>, tensor<f32>) -> tensor<256xf32>
      mhlo.return %379 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>) -> tensor<256xf32>
    %362 = "mhlo.fusion"(%360, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256xf32>):
      %377 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %379 = stablehlo.multiply %377, %378 : tensor<1x256xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x256xf32>
      %382 = stablehlo.rsqrt %381 : tensor<1x256xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %385 = stablehlo.multiply %arg208, %384 : tensor<1x256x256xf32>
      mhlo.return %385 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %363 = mhlo.bitcast %362 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
    %364 = "mhlo.fusion"(%363, %arg190, %arg191, %arg198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256x256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256xf32>) -> tensor<256x256xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : tensor<256x256xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %381 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %365 = "mhlo.fusion"(%364, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %378 = stablehlo.add %arg208, %377 : tensor<256x256xf32>
      mhlo.return %378 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<256x256xf32>
    %366 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %378 = stablehlo.iota dim = 0 : tensor<128xi32>
      %379 = stablehlo.convert %378 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %381 = stablehlo.add %379, %380 : tensor<128xf32>
      %382 = stablehlo.multiply %381, %380 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %384 = stablehlo.add %382, %383 : tensor<128xf32>
      %385 = stablehlo.maximum %377, %384 : tensor<128xf32>
      %386 = stablehlo.convert %385 : (tensor<128xf32>) -> tensor<128xi32>
      %387 = stablehlo.convert %386 : (tensor<128xi32>) -> tensor<128xf32>
      %388 = stablehlo.subtract %385, %387 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %390 = stablehlo.clamp %377, %388, %389 : tensor<128xf32>
      mhlo.return %390 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %367 = mhlo.bitcast %84#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<1x32x16384xf32>) -> tensor<16384x32xf32>
    %368 = "mhlo.fusion"(%367, %arg184, %arg185, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<256x32xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : (tensor<32xf32>) -> tensor<16384x32xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[16384,32]{0,1}"} : tensor<16384x32xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<256x32xf32>) -> tensor<16384x256xf32>
      mhlo.return %381 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<256x32xf32>) -> tensor<16384x256xf32>
    %369 = mhlo.bitcast %175#1 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<1x64x4096xf32>) -> tensor<4096x64xf32>
    %370 = "mhlo.fusion"(%369, %arg186, %arg187, %arg194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<256x64xf32>):
      %377 = stablehlo.broadcast_in_dim %arg209, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %378 = stablehlo.multiply %arg208, %377 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : (tensor<64xf32>) -> tensor<4096x64xf32>
      %380 = stablehlo.add %378, %379 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,64]{0,1}"} : tensor<4096x64xf32>
      %381 = stablehlo.dot_general %380, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
      mhlo.return %381 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %371 = "mhlo.fusion"(%1, %2, %3, %277, %278, %279, %280, %281, %365, %366, %368, %arg193, %370, %arg195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg208: tensor<128x1xf32>, %arg209: tensor<128x1xf32>, %arg210: tensor<128xf32>, %arg211: tensor<1024x256xf32>, %arg212: tensor<128x1xf32>, %arg213: tensor<128x1xf32>, %arg214: tensor<128xf32>, %arg215: tensor<128xf32>, %arg216: tensor<256x256xf32>, %arg217: tensor<128xf32>, %arg218: tensor<16384x256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<4096x256xf32>, %arg221: tensor<256xf32>):
      %377 = mhlo.bitcast %arg216 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<256x256xf32>) -> tensor<1x256x16x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %379 = stablehlo.iota dim = 0 : tensor<128xi32>
      %380 = stablehlo.convert %379 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %382 = stablehlo.add %380, %381 : tensor<128xf32>
      %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<128xf32>
      %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %386 = stablehlo.add %384, %385 : tensor<128xf32>
      %387 = stablehlo.maximum %378, %386 : tensor<128xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %389 = stablehlo.convert %388 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %390 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %391 = stablehlo.compare LT, %389, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %392 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %393 = stablehlo.add %389, %392 : tensor<128x1xi32>
      %394 = stablehlo.select %391, %393, %389 : tensor<128x1xi1>, tensor<128x1xi32>
      %395 = mhlo.bitcast %394 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %397 = stablehlo.convert %387 : (tensor<128xf32>) -> tensor<128xi32>
      %398 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %399 = stablehlo.compare LT, %397, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %400 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %401 = stablehlo.add %397, %400 : tensor<128xi32>
      %402 = stablehlo.select %399, %401, %397 : tensor<128xi1>, tensor<128xi32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %404 = stablehlo.concatenate %396, %403, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %406 = "stablehlo.gather"(%377, %405) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %407 = mhlo.bitcast %406 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %408 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %409 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %410 = stablehlo.add %397, %409 : tensor<128xi32>
      %411 = stablehlo.minimum %408, %410 : tensor<128xi32>
      %412 = stablehlo.compare LT, %411, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %413 = stablehlo.add %411, %400 : tensor<128xi32>
      %414 = stablehlo.select %412, %413, %411 : tensor<128xi1>, tensor<128xi32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %416 = stablehlo.concatenate %396, %415, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %418 = "stablehlo.gather"(%377, %417) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %420 = stablehlo.subtract %419, %407 : tensor<1x256x128x128xf32>
      %421 = stablehlo.broadcast_in_dim %arg215, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x256x128x128xf32>
      %423 = stablehlo.add %407, %422 : tensor<1x256x128x128xf32>
      %424 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %425 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %389, %425 : tensor<128x1xi32>
      %427 = stablehlo.minimum %424, %426 : tensor<128x1xi32>
      %428 = stablehlo.compare LT, %427, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %429 = stablehlo.add %427, %392 : tensor<128x1xi32>
      %430 = stablehlo.select %428, %429, %427 : tensor<128x1xi1>, tensor<128x1xi32>
      %431 = mhlo.bitcast %430 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %432 = stablehlo.broadcast_in_dim %431, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %433 = stablehlo.concatenate %432, %403, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %435 = "stablehlo.gather"(%377, %434) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %437 = stablehlo.concatenate %432, %415, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %439 = "stablehlo.gather"(%377, %438) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %441 = stablehlo.subtract %440, %436 : tensor<1x256x128x128xf32>
      %442 = stablehlo.multiply %441, %421 : tensor<1x256x128x128xf32>
      %443 = stablehlo.add %436, %442 : tensor<1x256x128x128xf32>
      %444 = stablehlo.subtract %443, %423 : tensor<1x256x128x128xf32>
      %445 = mhlo.bitcast %arg208 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %446 = stablehlo.broadcast_in_dim %445, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %447 = stablehlo.multiply %444, %446 : tensor<1x256x128x128xf32>
      %448 = stablehlo.add %423, %447 : tensor<1x256x128x128xf32>
      %449 = mhlo.bitcast %arg211 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,32,32]{1,3,2,0}"} : (tensor<1024x256xf32>) -> tensor<1x256x32x32xf32>
      %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %451 = stablehlo.multiply %382, %450 : tensor<128xf32>
      %452 = stablehlo.add %451, %385 : tensor<128xf32>
      %453 = stablehlo.maximum %378, %452 : tensor<128xf32>
      %454 = mhlo.bitcast %453 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %455 = stablehlo.convert %454 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %456 = stablehlo.compare LT, %455, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %457 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %458 = stablehlo.add %455, %457 : tensor<128x1xi32>
      %459 = stablehlo.select %456, %458, %455 : tensor<128x1xi1>, tensor<128x1xi32>
      %460 = mhlo.bitcast %459 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %461 = stablehlo.broadcast_in_dim %460, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %462 = stablehlo.convert %453 : (tensor<128xf32>) -> tensor<128xi32>
      %463 = stablehlo.compare LT, %462, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %464 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %465 = stablehlo.add %462, %464 : tensor<128xi32>
      %466 = stablehlo.select %463, %465, %462 : tensor<128xi1>, tensor<128xi32>
      %467 = stablehlo.broadcast_in_dim %466, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %468 = stablehlo.concatenate %461, %467, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %469 = mhlo.bitcast %468 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %470 = "stablehlo.gather"(%449, %469) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %c_8 = stablehlo.constant dense<31> : tensor<i32>
      %472 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %473 = stablehlo.add %462, %409 : tensor<128xi32>
      %474 = stablehlo.minimum %472, %473 : tensor<128xi32>
      %475 = stablehlo.compare LT, %474, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %476 = stablehlo.add %474, %464 : tensor<128xi32>
      %477 = stablehlo.select %475, %476, %474 : tensor<128xi1>, tensor<128xi32>
      %478 = stablehlo.broadcast_in_dim %477, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %479 = stablehlo.concatenate %461, %478, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %480 = mhlo.bitcast %479 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %481 = "stablehlo.gather"(%449, %480) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %482 = mhlo.bitcast %481 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %483 = stablehlo.subtract %482, %471 : tensor<1x256x128x128xf32>
      %484 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<1x256x128x128xf32>
      %486 = stablehlo.add %471, %485 : tensor<1x256x128x128xf32>
      %487 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %488 = stablehlo.add %455, %425 : tensor<128x1xi32>
      %489 = stablehlo.minimum %487, %488 : tensor<128x1xi32>
      %490 = stablehlo.compare LT, %489, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %491 = stablehlo.add %489, %457 : tensor<128x1xi32>
      %492 = stablehlo.select %490, %491, %489 : tensor<128x1xi1>, tensor<128x1xi32>
      %493 = mhlo.bitcast %492 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %494 = stablehlo.broadcast_in_dim %493, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %495 = stablehlo.concatenate %494, %467, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %496 = mhlo.bitcast %495 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %497 = "stablehlo.gather"(%449, %496) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %499 = stablehlo.concatenate %494, %478, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %500 = mhlo.bitcast %499 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %501 = "stablehlo.gather"(%449, %500) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %502 = mhlo.bitcast %501 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %503 = stablehlo.subtract %502, %498 : tensor<1x256x128x128xf32>
      %504 = stablehlo.multiply %503, %484 : tensor<1x256x128x128xf32>
      %505 = stablehlo.add %498, %504 : tensor<1x256x128x128xf32>
      %506 = stablehlo.subtract %505, %486 : tensor<1x256x128x128xf32>
      %507 = mhlo.bitcast %arg209 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %508 = stablehlo.broadcast_in_dim %507, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %509 = stablehlo.multiply %506, %508 : tensor<1x256x128x128xf32>
      %510 = stablehlo.add %486, %509 : tensor<1x256x128x128xf32>
      %511 = stablehlo.broadcast_in_dim %arg221, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %512 = stablehlo.add %arg220, %511 : tensor<4096x256xf32>
      %513 = mhlo.bitcast %512 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<4096x256xf32>) -> tensor<1x256x64x64xf32>
      %514 = stablehlo.multiply %382, %381 : tensor<128xf32>
      %515 = stablehlo.add %514, %385 : tensor<128xf32>
      %516 = stablehlo.maximum %378, %515 : tensor<128xf32>
      %517 = mhlo.bitcast %516 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %518 = stablehlo.convert %517 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %519 = stablehlo.compare LT, %518, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<64> : tensor<i32>
      %520 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %521 = stablehlo.add %518, %520 : tensor<128x1xi32>
      %522 = stablehlo.select %519, %521, %518 : tensor<128x1xi1>, tensor<128x1xi32>
      %523 = mhlo.bitcast %522 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %524 = stablehlo.broadcast_in_dim %523, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %525 = stablehlo.convert %516 : (tensor<128xf32>) -> tensor<128xi32>
      %526 = stablehlo.compare LT, %525, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %527 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %528 = stablehlo.add %525, %527 : tensor<128xi32>
      %529 = stablehlo.select %526, %528, %525 : tensor<128xi1>, tensor<128xi32>
      %530 = stablehlo.broadcast_in_dim %529, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %531 = stablehlo.concatenate %524, %530, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %532 = mhlo.bitcast %531 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %533 = "stablehlo.gather"(%513, %532) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %534 = mhlo.bitcast %533 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %c_10 = stablehlo.constant dense<63> : tensor<i32>
      %535 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %536 = stablehlo.add %525, %409 : tensor<128xi32>
      %537 = stablehlo.minimum %535, %536 : tensor<128xi32>
      %538 = stablehlo.compare LT, %537, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %539 = stablehlo.add %537, %527 : tensor<128xi32>
      %540 = stablehlo.select %538, %539, %537 : tensor<128xi1>, tensor<128xi32>
      %541 = stablehlo.broadcast_in_dim %540, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %542 = stablehlo.concatenate %524, %541, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %543 = mhlo.bitcast %542 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %544 = "stablehlo.gather"(%513, %543) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %545 = mhlo.bitcast %544 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %546 = stablehlo.subtract %545, %534 : tensor<1x256x128x128xf32>
      %547 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %548 = stablehlo.multiply %546, %547 : tensor<1x256x128x128xf32>
      %549 = stablehlo.add %534, %548 : tensor<1x256x128x128xf32>
      %550 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %551 = stablehlo.add %518, %425 : tensor<128x1xi32>
      %552 = stablehlo.minimum %550, %551 : tensor<128x1xi32>
      %553 = stablehlo.compare LT, %552, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %554 = stablehlo.add %552, %520 : tensor<128x1xi32>
      %555 = stablehlo.select %553, %554, %552 : tensor<128x1xi1>, tensor<128x1xi32>
      %556 = mhlo.bitcast %555 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %557 = stablehlo.broadcast_in_dim %556, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %558 = stablehlo.concatenate %557, %530, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %559 = mhlo.bitcast %558 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %560 = "stablehlo.gather"(%513, %559) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %561 = mhlo.bitcast %560 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %562 = stablehlo.concatenate %557, %541, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %563 = mhlo.bitcast %562 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %564 = "stablehlo.gather"(%513, %563) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %565 = mhlo.bitcast %564 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %566 = stablehlo.subtract %565, %561 : tensor<1x256x128x128xf32>
      %567 = stablehlo.multiply %566, %547 : tensor<1x256x128x128xf32>
      %568 = stablehlo.add %561, %567 : tensor<1x256x128x128xf32>
      %569 = stablehlo.subtract %568, %549 : tensor<1x256x128x128xf32>
      %570 = mhlo.bitcast %arg212 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %571 = stablehlo.broadcast_in_dim %570, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %572 = stablehlo.multiply %569, %571 : tensor<1x256x128x128xf32>
      %573 = stablehlo.add %549, %572 : tensor<1x256x128x128xf32>
      %574 = stablehlo.broadcast_in_dim %arg219, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %575 = stablehlo.add %arg218, %574 : tensor<16384x256xf32>
      %576 = mhlo.bitcast %575 {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[1,256,128,128]{1,3,2,0}"} : (tensor<16384x256xf32>) -> tensor<1x256x128x128xf32>
      %577 = stablehlo.maximum %378, %380 : tensor<128xf32>
      %578 = mhlo.bitcast %577 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %579 = stablehlo.convert %578 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %580 = stablehlo.compare LT, %579, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<128> : tensor<i32>
      %581 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %582 = stablehlo.add %579, %581 : tensor<128x1xi32>
      %583 = stablehlo.select %580, %582, %579 : tensor<128x1xi1>, tensor<128x1xi32>
      %584 = mhlo.bitcast %583 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %585 = stablehlo.broadcast_in_dim %584, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %586 = stablehlo.convert %577 : (tensor<128xf32>) -> tensor<128xi32>
      %587 = stablehlo.compare LT, %586, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %588 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %589 = stablehlo.add %586, %588 : tensor<128xi32>
      %590 = stablehlo.select %587, %589, %586 : tensor<128xi1>, tensor<128xi32>
      %591 = stablehlo.broadcast_in_dim %590, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %592 = stablehlo.concatenate %585, %591, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %593 = mhlo.bitcast %592 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %594 = "stablehlo.gather"(%576, %593) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %595 = mhlo.bitcast %594 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %c_12 = stablehlo.constant dense<127> : tensor<i32>
      %596 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %597 = stablehlo.add %586, %409 : tensor<128xi32>
      %598 = stablehlo.minimum %596, %597 : tensor<128xi32>
      %599 = stablehlo.compare LT, %598, %398 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %600 = stablehlo.add %598, %588 : tensor<128xi32>
      %601 = stablehlo.select %599, %600, %598 : tensor<128xi1>, tensor<128xi32>
      %602 = stablehlo.broadcast_in_dim %601, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %603 = stablehlo.concatenate %585, %602, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %604 = mhlo.bitcast %603 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %605 = "stablehlo.gather"(%576, %604) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %606 = mhlo.bitcast %605 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %607 = stablehlo.subtract %606, %595 : tensor<1x256x128x128xf32>
      %608 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %609 = stablehlo.multiply %607, %608 : tensor<1x256x128x128xf32>
      %610 = stablehlo.add %595, %609 : tensor<1x256x128x128xf32>
      %611 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %612 = stablehlo.add %579, %425 : tensor<128x1xi32>
      %613 = stablehlo.minimum %611, %612 : tensor<128x1xi32>
      %614 = stablehlo.compare LT, %613, %390 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %615 = stablehlo.add %613, %581 : tensor<128x1xi32>
      %616 = stablehlo.select %614, %615, %613 : tensor<128x1xi1>, tensor<128x1xi32>
      %617 = mhlo.bitcast %616 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %618 = stablehlo.broadcast_in_dim %617, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %619 = stablehlo.concatenate %618, %591, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %620 = mhlo.bitcast %619 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %621 = "stablehlo.gather"(%576, %620) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %622 = mhlo.bitcast %621 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %623 = stablehlo.concatenate %618, %602, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %624 = mhlo.bitcast %623 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %625 = "stablehlo.gather"(%576, %624) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> {result_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>, xla_shape = "f32[16384,1,256,1,1]{0,2,4,3,1}"} : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %626 = mhlo.bitcast %625 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[0, 2, 4, 3, 1]> : tensor<5xindex>} : (tensor<16384x1x256x1x1xf32>) -> tensor<1x256x128x128xf32>
      %627 = stablehlo.subtract %626, %622 : tensor<1x256x128x128xf32>
      %628 = stablehlo.multiply %627, %608 : tensor<1x256x128x128xf32>
      %629 = stablehlo.add %622, %628 : tensor<1x256x128x128xf32>
      %630 = stablehlo.subtract %629, %610 : tensor<1x256x128x128xf32>
      %631 = mhlo.bitcast %arg213 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %632 = stablehlo.broadcast_in_dim %631, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %633 = stablehlo.multiply %630, %632 : tensor<1x256x128x128xf32>
      %634 = stablehlo.add %610, %633 : tensor<1x256x128x128xf32>
      %635 = stablehlo.concatenate %448, %510, %573, %634, dim = 1 : (tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x1024x128x128xf32>
      mhlo.return %635 : tensor<1x1024x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>, tensor<1024x256xf32>, tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x256xf32>, tensor<128xf32>, tensor<16384x256xf32>, tensor<256xf32>, tensor<4096x256xf32>, tensor<256xf32>) -> tensor<1x1024x128x128xf32>
    %372 = stablehlo.custom_call @__cudnn$convForward(%371, %arg200) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,256,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x1024x128x128xf32>, tensor<256x1024x1x1xf32>) -> tuple<tensor<1x256x128x128xf32>, tensor<0xui8>>
    %373 = stablehlo.get_tuple_element %372[0] : (tuple<tensor<1x256x128x128xf32>, tensor<0xui8>>) -> tensor<1x256x128x128xf32>
    %374 = "mhlo.fusion"(%arg202, %arg201, %0, %373, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1x256x128x128xf32>, %arg212: tensor<256xf32>):
      %377 = stablehlo.broadcast_in_dim %arg212, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
      %378 = stablehlo.subtract %arg211, %377 : tensor<1x256x128x128xf32>
      %379 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x256x128x128xf32>
      %381 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
      %382 = stablehlo.multiply %380, %381 : tensor<1x256x128x128xf32>
      %383 = stablehlo.broadcast_in_dim %arg208, dims = [1] : (tensor<256xf32>) -> tensor<1x256x128x128xf32>
      %384 = stablehlo.add %382, %383 : tensor<1x256x128x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x128x128xf32>
      %386 = stablehlo.maximum %384, %385 : tensor<1x256x128x128xf32>
      mhlo.return %386 : tensor<1x256x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x128x128xf32>, tensor<256xf32>) -> tensor<1x256x128x128xf32>
    %375 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%374, %arg203, %arg204) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,150,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1x256x128x128xf32>, tensor<150x256x1x1xf32>, tensor<150xf32>) -> tuple<tensor<1x150x128x128xf32>, tensor<0xui8>>
    %376 = stablehlo.get_tuple_element %375[0] : (tuple<tensor<1x150x128x128xf32>, tensor<0xui8>>) -> tensor<1x150x128x128xf32>
    return %376 : tensor<1x150x128x128xf32>
  }
}
