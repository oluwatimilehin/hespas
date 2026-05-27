module {
  func.func @main(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<768x3x16x16xf32>, %arg2: tensor<1x197x768xf32>, %arg3: tensor<1x1x768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768x768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<3072x768xf32>, %arg18: tensor<3072xf32>, %arg19: tensor<768x3072xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768x768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<3072x768xf32>, %arg34: tensor<3072xf32>, %arg35: tensor<768x3072xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<3072x768xf32>, %arg50: tensor<3072xf32>, %arg51: tensor<768x3072xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<3072x768xf32>, %arg66: tensor<3072xf32>, %arg67: tensor<768x3072xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<3072x768xf32>, %arg82: tensor<3072xf32>, %arg83: tensor<768x3072xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<3072x768xf32>, %arg98: tensor<3072xf32>, %arg99: tensor<768x3072xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768x768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<3072x768xf32>, %arg114: tensor<3072xf32>, %arg115: tensor<768x3072xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768x768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<3072x768xf32>, %arg130: tensor<3072xf32>, %arg131: tensor<768x3072xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<3072x768xf32>, %arg146: tensor<3072xf32>, %arg147: tensor<768x3072xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768x768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<3072x768xf32>, %arg162: tensor<3072xf32>, %arg163: tensor<768x3072xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768x768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<3072x768xf32>, %arg178: tensor<3072xf32>, %arg179: tensor<768x3072xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<3072x768xf32>, %arg194: tensor<3072xf32>, %arg195: tensor<768x3072xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<768xf32>, %arg199: tensor<1000x768xf32>, %arg200: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x3x224x224xf32>):
      %420 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %421 = mhlo.copy %420 : tensor<1x224x224x3xf32>
      mhlo.return %421 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x16x16xf32>):
      %420 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[16,16,3,768]{1,0,2,3}"} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
      %421 = mhlo.copy %420 : tensor<16x16x3x768xf32>
      mhlo.return %421 : tensor<16x16x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<16x16x3x768xf32>) -> tensor<1x14x14x768xf32>
    %3 = "mhlo.fusion"(%arg2, %arg3, %2, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x1x768xf32>, %arg203: tensor<1x14x14x768xf32>, %arg204: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<768xf32>) -> tensor<1x14x14x768xf32>
      %421 = stablehlo.add %arg203, %420 : tensor<1x14x14x768xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,14,14]{1,3,2,0}"} : (tensor<1x14x14x768xf32>) -> tensor<1x768x14x14xf32>
      %423 = mhlo.copy %422 : tensor<1x768x14x14xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,768]{1,2,0}"} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
      %426 = mhlo.copy %425 : tensor<1x196x768xf32>
      %427 = stablehlo.concatenate %arg202, %426, dim = 1 : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
      %428 = stablehlo.add %427, %arg201 : tensor<1x197x768xf32>
      mhlo.return %428 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<1x14x14x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %5 = "mhlo.fusion"(%3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %7 = "mhlo.fusion"(%6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %8 = "mhlo.fusion"(%arg5, %arg6, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %9 = stablehlo.dot_general %8, %arg7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %10 = "mhlo.fusion"(%9, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %11 = stablehlo.dot_general %8, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %12 = "mhlo.fusion"(%11, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %13 = stablehlo.dot_general %10, %12, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = "mhlo.fusion"(%14, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %16 = "mhlo.fusion"(%15, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %17 = "mhlo.fusion"(%16, %13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %18 = "mhlo.fusion"(%17, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %19 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %20 = "mhlo.fusion"(%19, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %21 = "mhlo.fusion"(%20, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %22 = "mhlo.fusion"(%17, %18, %21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %23 = stablehlo.dot_general %8, %arg11, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %24 = "mhlo.fusion"(%23, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %25 = stablehlo.dot_general %22, %24, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %26 = "mhlo.fusion"(%25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %27 = stablehlo.dot_general %26, %arg13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %28 = "mhlo.fusion"(%3, %27, %arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %29 = "mhlo.fusion"(%28, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %30 = "mhlo.fusion"(%28, %29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %31 = "mhlo.fusion"(%30, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %32 = "mhlo.fusion"(%31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %33 = "mhlo.fusion"(%arg15, %arg16, %32, %28, %29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %34 = stablehlo.dot_general %33, %arg17, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %35 = "mhlo.fusion"(%34, %arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %36 = stablehlo.dot_general %35, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %37 = "mhlo.fusion"(%28, %36, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %38 = "mhlo.fusion"(%37, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %39 = "mhlo.fusion"(%37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %40 = "mhlo.fusion"(%39, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %42 = "mhlo.fusion"(%arg21, %arg22, %41, %37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %43 = stablehlo.dot_general %42, %arg23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %44 = "mhlo.fusion"(%43, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %45 = stablehlo.dot_general %42, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %46 = "mhlo.fusion"(%45, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %47 = stablehlo.dot_general %44, %46, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %49 = "mhlo.fusion"(%48, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %50 = "mhlo.fusion"(%49, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %51 = "mhlo.fusion"(%50, %47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %52 = "mhlo.fusion"(%51, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %53 = "mhlo.fusion"(%47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %54 = "mhlo.fusion"(%53, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %55 = "mhlo.fusion"(%54, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %56 = "mhlo.fusion"(%51, %52, %55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %57 = stablehlo.dot_general %42, %arg27, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %58 = "mhlo.fusion"(%57, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %59 = stablehlo.dot_general %56, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %60 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %61 = stablehlo.dot_general %60, %arg29, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %62 = "mhlo.fusion"(%37, %61, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %63 = "mhlo.fusion"(%62, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %64 = "mhlo.fusion"(%62, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %65 = "mhlo.fusion"(%64, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %66 = "mhlo.fusion"(%65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %67 = "mhlo.fusion"(%arg31, %arg32, %66, %62, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %68 = stablehlo.dot_general %67, %arg33, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %69 = "mhlo.fusion"(%68, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %70 = stablehlo.dot_general %69, %arg35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %71 = "mhlo.fusion"(%62, %70, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %72 = "mhlo.fusion"(%71, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %74 = "mhlo.fusion"(%73, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %76 = "mhlo.fusion"(%arg37, %arg38, %75, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %77 = stablehlo.dot_general %76, %arg39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %78 = "mhlo.fusion"(%77, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %79 = stablehlo.dot_general %76, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %80 = "mhlo.fusion"(%79, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %81 = stablehlo.dot_general %78, %80, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %83 = "mhlo.fusion"(%82, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %84 = "mhlo.fusion"(%83, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %85 = "mhlo.fusion"(%84, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %87 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %88 = "mhlo.fusion"(%87, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %89 = "mhlo.fusion"(%88, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %90 = "mhlo.fusion"(%85, %86, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %91 = stablehlo.dot_general %76, %arg43, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %92 = "mhlo.fusion"(%91, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %95 = stablehlo.dot_general %94, %arg45, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %96 = "mhlo.fusion"(%71, %95, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %97 = "mhlo.fusion"(%96, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %98 = "mhlo.fusion"(%96, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %99 = "mhlo.fusion"(%98, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %100 = "mhlo.fusion"(%99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %101 = "mhlo.fusion"(%arg47, %arg48, %100, %96, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %102 = stablehlo.dot_general %101, %arg49, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %103 = "mhlo.fusion"(%102, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %104 = stablehlo.dot_general %103, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %105 = "mhlo.fusion"(%96, %104, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %107 = "mhlo.fusion"(%105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %110 = "mhlo.fusion"(%arg53, %arg54, %109, %105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %111 = stablehlo.dot_general %110, %arg55, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %112 = "mhlo.fusion"(%111, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %113 = stablehlo.dot_general %110, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %114 = "mhlo.fusion"(%113, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %115 = stablehlo.dot_general %112, %114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %117 = "mhlo.fusion"(%116, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %118 = "mhlo.fusion"(%117, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %119 = "mhlo.fusion"(%118, %115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %120 = "mhlo.fusion"(%119, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %121 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %122 = "mhlo.fusion"(%121, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %123 = "mhlo.fusion"(%122, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %124 = "mhlo.fusion"(%119, %120, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %125 = stablehlo.dot_general %110, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %126 = "mhlo.fusion"(%125, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %127 = stablehlo.dot_general %124, %126, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %128 = "mhlo.fusion"(%127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %129 = stablehlo.dot_general %128, %arg61, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %130 = "mhlo.fusion"(%105, %129, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %131 = "mhlo.fusion"(%130, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %132 = "mhlo.fusion"(%130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %135 = "mhlo.fusion"(%arg63, %arg64, %134, %130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %136 = stablehlo.dot_general %135, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %137 = "mhlo.fusion"(%136, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %138 = stablehlo.dot_general %137, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %139 = "mhlo.fusion"(%130, %138, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %140 = "mhlo.fusion"(%139, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %141 = "mhlo.fusion"(%139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %144 = "mhlo.fusion"(%arg69, %arg70, %143, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %145 = stablehlo.dot_general %144, %arg71, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %146 = "mhlo.fusion"(%145, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %147 = stablehlo.dot_general %144, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %148 = "mhlo.fusion"(%147, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %149 = stablehlo.dot_general %146, %148, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %151 = "mhlo.fusion"(%150, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %152 = "mhlo.fusion"(%151, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %153 = "mhlo.fusion"(%152, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %154 = "mhlo.fusion"(%153, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %155 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %156 = "mhlo.fusion"(%155, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %157 = "mhlo.fusion"(%156, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %158 = "mhlo.fusion"(%153, %154, %157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %159 = stablehlo.dot_general %144, %arg75, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %160 = "mhlo.fusion"(%159, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %161 = stablehlo.dot_general %158, %160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %163 = stablehlo.dot_general %162, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %164 = "mhlo.fusion"(%139, %163, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %165 = "mhlo.fusion"(%164, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %166 = "mhlo.fusion"(%164, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %167 = "mhlo.fusion"(%166, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %169 = "mhlo.fusion"(%arg79, %arg80, %168, %164, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %170 = stablehlo.dot_general %169, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %171 = "mhlo.fusion"(%170, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %172 = stablehlo.dot_general %171, %arg83, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %173 = "mhlo.fusion"(%164, %172, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %174 = "mhlo.fusion"(%173, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %175 = "mhlo.fusion"(%173, %174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %176 = "mhlo.fusion"(%175, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %178 = "mhlo.fusion"(%arg85, %arg86, %177, %173, %174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %179 = stablehlo.dot_general %178, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %180 = "mhlo.fusion"(%179, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %181 = stablehlo.dot_general %178, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %182 = "mhlo.fusion"(%181, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %183 = stablehlo.dot_general %180, %182, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %185 = "mhlo.fusion"(%184, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %186 = "mhlo.fusion"(%185, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %187 = "mhlo.fusion"(%186, %183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %188 = "mhlo.fusion"(%187, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %189 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %190 = "mhlo.fusion"(%189, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %191 = "mhlo.fusion"(%190, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %192 = "mhlo.fusion"(%187, %188, %191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %193 = stablehlo.dot_general %178, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %194 = "mhlo.fusion"(%193, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %195 = stablehlo.dot_general %192, %194, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %197 = stablehlo.dot_general %196, %arg93, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %198 = "mhlo.fusion"(%173, %197, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %200 = "mhlo.fusion"(%198, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %201 = "mhlo.fusion"(%200, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %202 = "mhlo.fusion"(%201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %203 = "mhlo.fusion"(%arg95, %arg96, %202, %198, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %204 = stablehlo.dot_general %203, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %205 = "mhlo.fusion"(%204, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %206 = stablehlo.dot_general %205, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %207 = "mhlo.fusion"(%198, %206, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %208 = "mhlo.fusion"(%207, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %209 = "mhlo.fusion"(%207, %208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %212 = "mhlo.fusion"(%arg101, %arg102, %211, %207, %208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %213 = stablehlo.dot_general %212, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %214 = "mhlo.fusion"(%213, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %215 = stablehlo.dot_general %212, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %216 = "mhlo.fusion"(%215, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %217 = stablehlo.dot_general %214, %216, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %219 = "mhlo.fusion"(%218, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %220 = "mhlo.fusion"(%219, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %221 = "mhlo.fusion"(%220, %217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %222 = "mhlo.fusion"(%221, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %223 = "mhlo.fusion"(%217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %224 = "mhlo.fusion"(%223, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %225 = "mhlo.fusion"(%224, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %226 = "mhlo.fusion"(%221, %222, %225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %227 = stablehlo.dot_general %212, %arg107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %228 = "mhlo.fusion"(%227, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %229 = stablehlo.dot_general %226, %228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %231 = stablehlo.dot_general %230, %arg109, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %232 = "mhlo.fusion"(%207, %231, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %233 = "mhlo.fusion"(%232, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %234 = "mhlo.fusion"(%232, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %235 = "mhlo.fusion"(%234, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %236 = "mhlo.fusion"(%235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %237 = "mhlo.fusion"(%arg111, %arg112, %236, %232, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %238 = stablehlo.dot_general %237, %arg113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %239 = "mhlo.fusion"(%238, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %240 = stablehlo.dot_general %239, %arg115, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %241 = "mhlo.fusion"(%232, %240, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %242 = "mhlo.fusion"(%241, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %243 = "mhlo.fusion"(%241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %244 = "mhlo.fusion"(%243, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %246 = "mhlo.fusion"(%arg117, %arg118, %245, %241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %247 = stablehlo.dot_general %246, %arg119, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %248 = "mhlo.fusion"(%247, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %249 = stablehlo.dot_general %246, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %250 = "mhlo.fusion"(%249, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %251 = stablehlo.dot_general %248, %250, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %253 = "mhlo.fusion"(%252, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %254 = "mhlo.fusion"(%253, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %255 = "mhlo.fusion"(%254, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %257 = "mhlo.fusion"(%251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %258 = "mhlo.fusion"(%257, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %259 = "mhlo.fusion"(%258, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %260 = "mhlo.fusion"(%255, %256, %259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %261 = stablehlo.dot_general %246, %arg123, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %262 = "mhlo.fusion"(%261, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %263 = stablehlo.dot_general %260, %262, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %265 = stablehlo.dot_general %264, %arg125, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %266 = "mhlo.fusion"(%241, %265, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %268 = "mhlo.fusion"(%266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %269 = "mhlo.fusion"(%268, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %271 = "mhlo.fusion"(%arg127, %arg128, %270, %266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %272 = stablehlo.dot_general %271, %arg129, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %273 = "mhlo.fusion"(%272, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %274 = stablehlo.dot_general %273, %arg131, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %275 = "mhlo.fusion"(%266, %274, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %277 = "mhlo.fusion"(%275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %278 = "mhlo.fusion"(%277, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %279 = "mhlo.fusion"(%278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %280 = "mhlo.fusion"(%arg133, %arg134, %279, %275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %281 = stablehlo.dot_general %280, %arg135, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %282 = "mhlo.fusion"(%281, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %283 = stablehlo.dot_general %280, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %284 = "mhlo.fusion"(%283, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %285 = stablehlo.dot_general %282, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %287 = "mhlo.fusion"(%286, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %288 = "mhlo.fusion"(%287, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %289 = "mhlo.fusion"(%288, %285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %290 = "mhlo.fusion"(%289, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %291 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %292 = "mhlo.fusion"(%291, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %293 = "mhlo.fusion"(%292, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %294 = "mhlo.fusion"(%289, %290, %293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %295 = stablehlo.dot_general %280, %arg139, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %296 = "mhlo.fusion"(%295, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %297 = stablehlo.dot_general %294, %296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %298 = "mhlo.fusion"(%297) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %299 = stablehlo.dot_general %298, %arg141, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %300 = "mhlo.fusion"(%275, %299, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %302 = "mhlo.fusion"(%300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %304 = "mhlo.fusion"(%303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %305 = "mhlo.fusion"(%arg143, %arg144, %304, %300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %306 = stablehlo.dot_general %305, %arg145, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %307 = "mhlo.fusion"(%306, %arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %308 = stablehlo.dot_general %307, %arg147, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %309 = "mhlo.fusion"(%300, %308, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %311 = "mhlo.fusion"(%309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %313 = "mhlo.fusion"(%312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %314 = "mhlo.fusion"(%arg149, %arg150, %313, %309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %315 = stablehlo.dot_general %314, %arg151, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %316 = "mhlo.fusion"(%315, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %317 = stablehlo.dot_general %314, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %318 = "mhlo.fusion"(%317, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %319 = stablehlo.dot_general %316, %318, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %320 = mhlo.bitcast %319 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %322 = "mhlo.fusion"(%321, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %323 = "mhlo.fusion"(%322, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %324 = "mhlo.fusion"(%323, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %325 = "mhlo.fusion"(%319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %326 = "mhlo.fusion"(%325, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %327 = "mhlo.fusion"(%326, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %328 = "mhlo.fusion"(%323, %324, %327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %329 = stablehlo.dot_general %314, %arg155, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %330 = "mhlo.fusion"(%329, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %331 = stablehlo.dot_general %328, %330, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %332 = "mhlo.fusion"(%331) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %333 = stablehlo.dot_general %332, %arg157, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %334 = "mhlo.fusion"(%309, %333, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %335 = "mhlo.fusion"(%334, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %336 = "mhlo.fusion"(%334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %339 = "mhlo.fusion"(%arg159, %arg160, %338, %334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %340 = stablehlo.dot_general %339, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %341 = "mhlo.fusion"(%340, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %342 = stablehlo.dot_general %341, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %343 = "mhlo.fusion"(%334, %342, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %344 = "mhlo.fusion"(%343, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %345 = "mhlo.fusion"(%343, %344) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %346 = "mhlo.fusion"(%345, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %348 = "mhlo.fusion"(%arg165, %arg166, %347, %343, %344) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %349 = stablehlo.dot_general %348, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %350 = "mhlo.fusion"(%349, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %351 = stablehlo.dot_general %348, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %352 = "mhlo.fusion"(%351, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %353 = stablehlo.dot_general %350, %352, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %354 = mhlo.bitcast %353 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %355 = "mhlo.fusion"(%354, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %356 = "mhlo.fusion"(%355, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %357 = "mhlo.fusion"(%356, %353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %358 = "mhlo.fusion"(%357, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %359 = "mhlo.fusion"(%353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %360 = "mhlo.fusion"(%359, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %361 = "mhlo.fusion"(%360, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %362 = "mhlo.fusion"(%357, %358, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %363 = stablehlo.dot_general %348, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %364 = "mhlo.fusion"(%363, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %365 = stablehlo.dot_general %362, %364, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %366 = "mhlo.fusion"(%365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %367 = stablehlo.dot_general %366, %arg173, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %368 = "mhlo.fusion"(%343, %367, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %369 = "mhlo.fusion"(%368, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %370 = "mhlo.fusion"(%368, %369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %371 = "mhlo.fusion"(%370, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %372 = "mhlo.fusion"(%371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %373 = "mhlo.fusion"(%arg175, %arg176, %372, %368, %369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %374 = stablehlo.dot_general %373, %arg177, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %375 = "mhlo.fusion"(%374, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %376 = stablehlo.dot_general %375, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %377 = "mhlo.fusion"(%368, %376, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %379 = "mhlo.fusion"(%377, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %380 = "mhlo.fusion"(%379, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %381 = "mhlo.fusion"(%380) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %382 = "mhlo.fusion"(%arg181, %arg182, %381, %377, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %383 = stablehlo.dot_general %382, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %384 = "mhlo.fusion"(%383, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %385 = stablehlo.dot_general %382, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %386 = "mhlo.fusion"(%385, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %387 = stablehlo.dot_general %384, %386, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %389 = "mhlo.fusion"(%388, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %390 = "mhlo.fusion"(%389, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %391 = "mhlo.fusion"(%390, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %392 = "mhlo.fusion"(%391, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %393 = "mhlo.fusion"(%387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %394 = "mhlo.fusion"(%393, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %395 = "mhlo.fusion"(%394, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %396 = "mhlo.fusion"(%391, %392, %395) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %397 = stablehlo.dot_general %382, %arg187, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %398 = "mhlo.fusion"(%397, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %399 = stablehlo.dot_general %396, %398, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %400 = "mhlo.fusion"(%399) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %401 = stablehlo.dot_general %400, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %402 = "mhlo.fusion"(%377, %401, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %403 = "mhlo.fusion"(%402, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %404 = "mhlo.fusion"(%402, %403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %405 = "mhlo.fusion"(%404, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %406 = "mhlo.fusion"(%405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %407 = "mhlo.fusion"(%arg191, %arg192, %406, %402, %403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %408 = stablehlo.dot_general %407, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %409 = "mhlo.fusion"(%408, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %410 = stablehlo.dot_general %409, %arg195, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %411 = "mhlo.fusion"(%402, %410, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %412 = "mhlo.fusion"(%411, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %413 = "mhlo.fusion"(%411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %414 = "mhlo.fusion"(%413, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %415 = "mhlo.fusion"(%414) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %416 = "mhlo.fusion"(%arg197, %arg198, %415, %411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = stablehlo.slice %431 [0:1, 0:1, 0:768] : (tensor<1x197x768xf32>) -> tensor<1x1x768xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      mhlo.return %433 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<768xf32>
    %417 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1000x768xf32>):
      %420 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,1000]{0,1}"} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
      %421 = mhlo.copy %420 : tensor<768x1000xf32>
      mhlo.return %421 : tensor<768x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
    %418 = "mhlo.fusion"(%arg200, %416, %417) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg201: tensor<1000xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768x1000xf32>):
      %420 = stablehlo.dot %arg202, %arg203, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
      %421 = stablehlo.add %arg201, %420 : tensor<1000xf32>
      mhlo.return %421 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
    %419 = mhlo.bitcast %418 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %419 : tensor<1x1000xf32>
  }
}
