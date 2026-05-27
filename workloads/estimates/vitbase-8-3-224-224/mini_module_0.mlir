module {
  func.func @main(%arg0: tensor<8x3x224x224xf32>, %arg1: tensor<768x3x16x16xf32>, %arg2: tensor<768xf32>, %arg3: tensor<1x1x768xf32>, %arg4: tensor<1x197x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768x768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<3072x768xf32>, %arg18: tensor<3072xf32>, %arg19: tensor<768x3072xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768x768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<3072x768xf32>, %arg34: tensor<3072xf32>, %arg35: tensor<768x3072xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<3072x768xf32>, %arg50: tensor<3072xf32>, %arg51: tensor<768x3072xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<3072x768xf32>, %arg66: tensor<3072xf32>, %arg67: tensor<768x3072xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<3072x768xf32>, %arg82: tensor<3072xf32>, %arg83: tensor<768x3072xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<3072x768xf32>, %arg98: tensor<3072xf32>, %arg99: tensor<768x3072xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768x768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<3072x768xf32>, %arg114: tensor<3072xf32>, %arg115: tensor<768x3072xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768x768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<3072x768xf32>, %arg130: tensor<3072xf32>, %arg131: tensor<768x3072xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<3072x768xf32>, %arg146: tensor<3072xf32>, %arg147: tensor<768x3072xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768x768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<3072x768xf32>, %arg162: tensor<3072xf32>, %arg163: tensor<768x3072xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768x768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<3072x768xf32>, %arg178: tensor<3072xf32>, %arg179: tensor<768x3072xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<3072x768xf32>, %arg194: tensor<3072xf32>, %arg195: tensor<768x3072xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<768xf32>, %arg199: tensor<1000x768xf32>, %arg200: tensor<1000xf32>) -> tensor<8x1000xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x3x224x224xf32>):
      %457 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,224,224,3]{2,1,3,0}"} : (tensor<8x3x224x224xf32>) -> tensor<8x224x224x3xf32>
      %458 = mhlo.copy %457 : tensor<8x224x224x3xf32>
      mhlo.return %458 : tensor<8x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3x224x224xf32>) -> tensor<8x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x16x16xf32>):
      %457 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[16,16,3,768]{1,0,2,3}"} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
      %458 = mhlo.copy %457 : tensor<16x16x3x768xf32>
      mhlo.return %458 : tensor<16x16x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x224x224x3xf32>, tensor<16x16x3x768xf32>) -> tensor<8x14x14x768xf32>
    %3 = "mhlo.fusion"(%2, %arg2, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x1x768xf32>):
      %457 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [2] : (tensor<768xf32>) -> tensor<8x1x768xf32>
      %459 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x14x14x768xf32>
      %460 = stablehlo.add %arg201, %459 : tensor<8x14x14x768xf32>
      %461 = stablehlo.transpose %460, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,768,14,14]{1,3,2,0}"} : (tensor<8x14x14x768xf32>) -> tensor<8x768x14x14xf32>
      %462 = mhlo.copy %461 : tensor<8x768x14x14xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x768x14x14xf32>) -> tensor<8x768x196xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,196,768]{1,2,0}"} : (tensor<8x768x196xf32>) -> tensor<8x196x768xf32>
      %465 = mhlo.copy %464 : tensor<8x196x768xf32>
      %466 = stablehlo.concatenate %458, %465, dim = 1 : (tensor<8x1x768xf32>, tensor<8x196x768xf32>) -> tensor<8x197x768xf32>
      mhlo.return %466 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x768xf32>, tensor<768xf32>, tensor<1x1x768xf32>) -> tensor<8x197x768xf32>
    %4 = "mhlo.fusion"(%3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1x197x768xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<8x197x768xf32>
      %459 = stablehlo.add %arg201, %458 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1x197x768xf32>) -> tensor<8x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "mhlo.fusion"(%4, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %6 = "mhlo.fusion"(%5, %3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<1x197x768xf32>):
      %457 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<8x197x768xf32>
      %459 = stablehlo.add %arg202, %458 : tensor<8x197x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %461 = stablehlo.multiply %arg201, %460 : tensor<8x197xf32>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %463 = stablehlo.subtract %459, %462 : tensor<8x197x768xf32>
      %464 = stablehlo.multiply %463, %463 : tensor<8x197x768xf32>
      mhlo.return %464 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<1x197x768xf32>) -> tensor<8x197x768xf32>
    %7 = "mhlo.fusion"(%6, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %8 = "mhlo.fusion"(%7, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %9 = "mhlo.fusion"(%8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %10 = "mhlo.fusion"(%arg5, %arg6, %9, %5, %3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197xf32>, %arg205: tensor<8x197x768xf32>, %arg206: tensor<1x197x768xf32>):
      %457 = mhlo.bitcast %arg206 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<8x197x768xf32>
      %459 = stablehlo.add %arg205, %458 : tensor<8x197x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %461 = stablehlo.multiply %arg204, %460 : tensor<8x197xf32>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %463 = stablehlo.subtract %459, %462 : tensor<8x197x768xf32>
      %464 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %465 = stablehlo.multiply %463, %464 : tensor<8x197x768xf32>
      %466 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<8x197x768xf32>
      %468 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x197x768xf32>
      %470 = mhlo.bitcast %469 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %470 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1576x768xf32>
    %11 = stablehlo.dot_general %10, %arg7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %12 = "mhlo.fusion"(%11, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %13 = stablehlo.dot_general %10, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %14 = "mhlo.fusion"(%13, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %15 = stablehlo.dot_general %12, %14, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = "mhlo.fusion"(%16, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %18 = "mhlo.fusion"(%17, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %19 = "mhlo.fusion"(%18, %15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %20 = "mhlo.fusion"(%19, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %21 = "mhlo.fusion"(%15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %22 = "mhlo.fusion"(%21, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %23 = "mhlo.fusion"(%22, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %24 = "mhlo.fusion"(%19, %20, %23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %25 = stablehlo.dot_general %10, %arg11, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %26 = "mhlo.fusion"(%25, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %27 = stablehlo.dot_general %24, %26, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %28 = "mhlo.fusion"(%27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %29 = "mhlo.fusion"(%arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %30 = stablehlo.dot_general %28, %29, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %31 = "mhlo.fusion"(%30, %arg14, %3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197x768xf32>, %arg204: tensor<1x197x768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg201, %457 : tensor<8x197x768xf32>
      %459 = mhlo.bitcast %arg204 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [1, 2] : (tensor<197x768xf32>) -> tensor<8x197x768xf32>
      %461 = stablehlo.add %arg203, %460 : tensor<8x197x768xf32>
      %462 = stablehlo.add %458, %461 : tensor<8x197x768xf32>
      mhlo.return %462 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<768xf32>, tensor<8x197x768xf32>, tensor<1x197x768xf32>) -> tensor<8x197x768xf32>
    %32 = "mhlo.fusion"(%31, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %33 = "mhlo.fusion"(%31, %32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %34 = "mhlo.fusion"(%33, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %36 = "mhlo.fusion"(%35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %37 = "mhlo.fusion"(%arg15, %arg16, %36, %31, %32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %38 = stablehlo.dot_general %37, %arg17, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %39 = "mhlo.fusion"(%38, %arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %40 = stablehlo.dot_general %39, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %41 = "mhlo.fusion"(%31, %40, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %42 = "mhlo.fusion"(%41, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %43 = "mhlo.fusion"(%41, %42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %44 = "mhlo.fusion"(%43, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %45 = "mhlo.fusion"(%44, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %46 = "mhlo.fusion"(%45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %47 = "mhlo.fusion"(%arg21, %arg22, %46, %41, %42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %48 = stablehlo.dot_general %47, %arg23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %49 = "mhlo.fusion"(%48, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %50 = stablehlo.dot_general %47, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %51 = "mhlo.fusion"(%50, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %52 = stablehlo.dot_general %49, %51, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %53 = mhlo.bitcast %52 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %54 = "mhlo.fusion"(%53, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %55 = "mhlo.fusion"(%54, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %56 = "mhlo.fusion"(%55, %52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %57 = "mhlo.fusion"(%56, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %58 = "mhlo.fusion"(%52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %59 = "mhlo.fusion"(%58, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %60 = "mhlo.fusion"(%59, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %61 = "mhlo.fusion"(%56, %57, %60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %62 = stablehlo.dot_general %47, %arg27, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %63 = "mhlo.fusion"(%62, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %64 = stablehlo.dot_general %61, %63, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %65 = "mhlo.fusion"(%64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %66 = "mhlo.fusion"(%arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %67 = stablehlo.dot_general %65, %66, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %68 = "mhlo.fusion"(%41, %67, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %69 = "mhlo.fusion"(%68, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %70 = "mhlo.fusion"(%68, %69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %71 = "mhlo.fusion"(%70, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %72 = "mhlo.fusion"(%71, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %74 = "mhlo.fusion"(%arg31, %arg32, %73, %68, %69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %75 = stablehlo.dot_general %74, %arg33, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %76 = "mhlo.fusion"(%75, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %77 = stablehlo.dot_general %76, %arg35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %78 = "mhlo.fusion"(%68, %77, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %79 = "mhlo.fusion"(%78, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %80 = "mhlo.fusion"(%78, %79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %81 = "mhlo.fusion"(%80, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %82 = "mhlo.fusion"(%81, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %84 = "mhlo.fusion"(%arg37, %arg38, %83, %78, %79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %85 = stablehlo.dot_general %84, %arg39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %86 = "mhlo.fusion"(%85, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %87 = stablehlo.dot_general %84, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %88 = "mhlo.fusion"(%87, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %89 = stablehlo.dot_general %86, %88, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %90 = mhlo.bitcast %89 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %91 = "mhlo.fusion"(%90, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %92 = "mhlo.fusion"(%91, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %93 = "mhlo.fusion"(%92, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %94 = "mhlo.fusion"(%93, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %95 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %96 = "mhlo.fusion"(%95, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %97 = "mhlo.fusion"(%96, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %98 = "mhlo.fusion"(%93, %94, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %99 = stablehlo.dot_general %84, %arg43, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %100 = "mhlo.fusion"(%99, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %101 = stablehlo.dot_general %98, %100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %102 = "mhlo.fusion"(%101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %103 = "mhlo.fusion"(%arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %104 = stablehlo.dot_general %102, %103, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %105 = "mhlo.fusion"(%78, %104, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %107 = "mhlo.fusion"(%105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %110 = "mhlo.fusion"(%109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %111 = "mhlo.fusion"(%arg47, %arg48, %110, %105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %112 = stablehlo.dot_general %111, %arg49, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %113 = "mhlo.fusion"(%112, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %114 = stablehlo.dot_general %113, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %115 = "mhlo.fusion"(%105, %114, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %117 = "mhlo.fusion"(%115, %116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %118 = "mhlo.fusion"(%117, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %119 = "mhlo.fusion"(%118, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %121 = "mhlo.fusion"(%arg53, %arg54, %120, %115, %116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %122 = stablehlo.dot_general %121, %arg55, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %123 = "mhlo.fusion"(%122, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %124 = stablehlo.dot_general %121, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %125 = "mhlo.fusion"(%124, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %126 = stablehlo.dot_general %123, %125, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %127 = mhlo.bitcast %126 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %128 = "mhlo.fusion"(%127, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %129 = "mhlo.fusion"(%128, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %130 = "mhlo.fusion"(%129, %126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %131 = "mhlo.fusion"(%130, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %132 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %133 = "mhlo.fusion"(%132, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %134 = "mhlo.fusion"(%133, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %135 = "mhlo.fusion"(%130, %131, %134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %136 = stablehlo.dot_general %121, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %137 = "mhlo.fusion"(%136, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %138 = stablehlo.dot_general %135, %137, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %140 = "mhlo.fusion"(%arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %141 = stablehlo.dot_general %139, %140, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %142 = "mhlo.fusion"(%115, %141, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %143 = "mhlo.fusion"(%142, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %144 = "mhlo.fusion"(%142, %143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %145 = "mhlo.fusion"(%144, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %146 = "mhlo.fusion"(%145, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %147 = "mhlo.fusion"(%146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %148 = "mhlo.fusion"(%arg63, %arg64, %147, %142, %143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %149 = stablehlo.dot_general %148, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %150 = "mhlo.fusion"(%149, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %151 = stablehlo.dot_general %150, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %152 = "mhlo.fusion"(%142, %151, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %154 = "mhlo.fusion"(%152, %153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %155 = "mhlo.fusion"(%154, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %156 = "mhlo.fusion"(%155, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %157 = "mhlo.fusion"(%156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %158 = "mhlo.fusion"(%arg69, %arg70, %157, %152, %153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %159 = stablehlo.dot_general %158, %arg71, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %160 = "mhlo.fusion"(%159, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %161 = stablehlo.dot_general %158, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %162 = "mhlo.fusion"(%161, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %163 = stablehlo.dot_general %160, %162, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %165 = "mhlo.fusion"(%164, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %166 = "mhlo.fusion"(%165, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %167 = "mhlo.fusion"(%166, %163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %168 = "mhlo.fusion"(%167, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %169 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %170 = "mhlo.fusion"(%169, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %171 = "mhlo.fusion"(%170, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %172 = "mhlo.fusion"(%167, %168, %171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %173 = stablehlo.dot_general %158, %arg75, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %174 = "mhlo.fusion"(%173, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %175 = stablehlo.dot_general %172, %174, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %176 = "mhlo.fusion"(%175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %177 = "mhlo.fusion"(%arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %178 = stablehlo.dot_general %176, %177, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %179 = "mhlo.fusion"(%152, %178, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %180 = "mhlo.fusion"(%179, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %181 = "mhlo.fusion"(%179, %180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %182 = "mhlo.fusion"(%181, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %183 = "mhlo.fusion"(%182, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %184 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %185 = "mhlo.fusion"(%arg79, %arg80, %184, %179, %180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %186 = stablehlo.dot_general %185, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %187 = "mhlo.fusion"(%186, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %188 = stablehlo.dot_general %187, %arg83, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %189 = "mhlo.fusion"(%179, %188, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %190 = "mhlo.fusion"(%189, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %191 = "mhlo.fusion"(%189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %192 = "mhlo.fusion"(%191, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %193 = "mhlo.fusion"(%192, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %195 = "mhlo.fusion"(%arg85, %arg86, %194, %189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %196 = stablehlo.dot_general %195, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %197 = "mhlo.fusion"(%196, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %198 = stablehlo.dot_general %195, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %199 = "mhlo.fusion"(%198, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %200 = stablehlo.dot_general %197, %199, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %202 = "mhlo.fusion"(%201, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %203 = "mhlo.fusion"(%202, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %204 = "mhlo.fusion"(%203, %200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %205 = "mhlo.fusion"(%204, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %206 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %207 = "mhlo.fusion"(%206, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %208 = "mhlo.fusion"(%207, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %209 = "mhlo.fusion"(%204, %205, %208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %210 = stablehlo.dot_general %195, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %211 = "mhlo.fusion"(%210, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %212 = stablehlo.dot_general %209, %211, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %213 = "mhlo.fusion"(%212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %214 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %215 = stablehlo.dot_general %213, %214, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %216 = "mhlo.fusion"(%189, %215, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %217 = "mhlo.fusion"(%216, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %218 = "mhlo.fusion"(%216, %217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %219 = "mhlo.fusion"(%218, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %220 = "mhlo.fusion"(%219, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %221 = "mhlo.fusion"(%220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %222 = "mhlo.fusion"(%arg95, %arg96, %221, %216, %217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %223 = stablehlo.dot_general %222, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %224 = "mhlo.fusion"(%223, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %225 = stablehlo.dot_general %224, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %226 = "mhlo.fusion"(%216, %225, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %227 = "mhlo.fusion"(%226, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %228 = "mhlo.fusion"(%226, %227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %229 = "mhlo.fusion"(%228, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %230 = "mhlo.fusion"(%229, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %232 = "mhlo.fusion"(%arg101, %arg102, %231, %226, %227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %233 = stablehlo.dot_general %232, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %234 = "mhlo.fusion"(%233, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %235 = stablehlo.dot_general %232, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %236 = "mhlo.fusion"(%235, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %237 = stablehlo.dot_general %234, %236, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %239 = "mhlo.fusion"(%238, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %240 = "mhlo.fusion"(%239, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %241 = "mhlo.fusion"(%240, %237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %242 = "mhlo.fusion"(%241, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %243 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %244 = "mhlo.fusion"(%243, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %245 = "mhlo.fusion"(%244, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %246 = "mhlo.fusion"(%241, %242, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %247 = stablehlo.dot_general %232, %arg107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %248 = "mhlo.fusion"(%247, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %249 = stablehlo.dot_general %246, %248, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %251 = "mhlo.fusion"(%arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %252 = stablehlo.dot_general %250, %251, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %253 = "mhlo.fusion"(%226, %252, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %254 = "mhlo.fusion"(%253, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %255 = "mhlo.fusion"(%253, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %257 = "mhlo.fusion"(%256, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %259 = "mhlo.fusion"(%arg111, %arg112, %258, %253, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %260 = stablehlo.dot_general %259, %arg113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %261 = "mhlo.fusion"(%260, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %262 = stablehlo.dot_general %261, %arg115, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %263 = "mhlo.fusion"(%253, %262, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %264 = "mhlo.fusion"(%263, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %265 = "mhlo.fusion"(%263, %264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %266 = "mhlo.fusion"(%265, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %269 = "mhlo.fusion"(%arg117, %arg118, %268, %263, %264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %270 = stablehlo.dot_general %269, %arg119, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %271 = "mhlo.fusion"(%270, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %272 = stablehlo.dot_general %269, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %273 = "mhlo.fusion"(%272, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %274 = stablehlo.dot_general %271, %273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %275 = mhlo.bitcast %274 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %276 = "mhlo.fusion"(%275, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %277 = "mhlo.fusion"(%276, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %278 = "mhlo.fusion"(%277, %274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %279 = "mhlo.fusion"(%278, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %280 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %281 = "mhlo.fusion"(%280, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %282 = "mhlo.fusion"(%281, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %283 = "mhlo.fusion"(%278, %279, %282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %284 = stablehlo.dot_general %269, %arg123, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %285 = "mhlo.fusion"(%284, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %286 = stablehlo.dot_general %283, %285, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %288 = "mhlo.fusion"(%arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %289 = stablehlo.dot_general %287, %288, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %290 = "mhlo.fusion"(%263, %289, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %291 = "mhlo.fusion"(%290, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %292 = "mhlo.fusion"(%290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %293 = "mhlo.fusion"(%292, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %294 = "mhlo.fusion"(%293, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %296 = "mhlo.fusion"(%arg127, %arg128, %295, %290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %297 = stablehlo.dot_general %296, %arg129, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %298 = "mhlo.fusion"(%297, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %299 = stablehlo.dot_general %298, %arg131, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %300 = "mhlo.fusion"(%290, %299, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %302 = "mhlo.fusion"(%300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %304 = "mhlo.fusion"(%303, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %305 = "mhlo.fusion"(%304) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %306 = "mhlo.fusion"(%arg133, %arg134, %305, %300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %307 = stablehlo.dot_general %306, %arg135, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %308 = "mhlo.fusion"(%307, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %309 = stablehlo.dot_general %306, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %310 = "mhlo.fusion"(%309, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %311 = stablehlo.dot_general %308, %310, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %312 = mhlo.bitcast %311 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %313 = "mhlo.fusion"(%312, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %314 = "mhlo.fusion"(%313, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %315 = "mhlo.fusion"(%314, %311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %316 = "mhlo.fusion"(%315, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %317 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %318 = "mhlo.fusion"(%317, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %319 = "mhlo.fusion"(%318, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %320 = "mhlo.fusion"(%315, %316, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %321 = stablehlo.dot_general %306, %arg139, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %322 = "mhlo.fusion"(%321, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %323 = stablehlo.dot_general %320, %322, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %324 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %325 = "mhlo.fusion"(%arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %326 = stablehlo.dot_general %324, %325, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %327 = "mhlo.fusion"(%300, %326, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %328 = "mhlo.fusion"(%327, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %329 = "mhlo.fusion"(%327, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %330 = "mhlo.fusion"(%329, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %331 = "mhlo.fusion"(%330, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %332 = "mhlo.fusion"(%331) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %333 = "mhlo.fusion"(%arg143, %arg144, %332, %327, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %334 = stablehlo.dot_general %333, %arg145, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %335 = "mhlo.fusion"(%334, %arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %336 = stablehlo.dot_general %335, %arg147, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %337 = "mhlo.fusion"(%327, %336, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %339 = "mhlo.fusion"(%337, %338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %340 = "mhlo.fusion"(%339, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %341 = "mhlo.fusion"(%340, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %342 = "mhlo.fusion"(%341) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %343 = "mhlo.fusion"(%arg149, %arg150, %342, %337, %338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %344 = stablehlo.dot_general %343, %arg151, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %345 = "mhlo.fusion"(%344, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %346 = stablehlo.dot_general %343, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %347 = "mhlo.fusion"(%346, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %348 = stablehlo.dot_general %345, %347, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %349 = mhlo.bitcast %348 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %350 = "mhlo.fusion"(%349, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %351 = "mhlo.fusion"(%350, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %352 = "mhlo.fusion"(%351, %348) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %353 = "mhlo.fusion"(%352, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %354 = "mhlo.fusion"(%348) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %355 = "mhlo.fusion"(%354, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %356 = "mhlo.fusion"(%355, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %357 = "mhlo.fusion"(%352, %353, %356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %358 = stablehlo.dot_general %343, %arg155, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %359 = "mhlo.fusion"(%358, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %360 = stablehlo.dot_general %357, %359, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %361 = "mhlo.fusion"(%360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %362 = "mhlo.fusion"(%arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %363 = stablehlo.dot_general %361, %362, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %364 = "mhlo.fusion"(%337, %363, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %365 = "mhlo.fusion"(%364, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %366 = "mhlo.fusion"(%364, %365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %367 = "mhlo.fusion"(%366, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %368 = "mhlo.fusion"(%367, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %369 = "mhlo.fusion"(%368) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %370 = "mhlo.fusion"(%arg159, %arg160, %369, %364, %365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %371 = stablehlo.dot_general %370, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %372 = "mhlo.fusion"(%371, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %373 = stablehlo.dot_general %372, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %374 = "mhlo.fusion"(%364, %373, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %376 = "mhlo.fusion"(%374, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %377 = "mhlo.fusion"(%376, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %379 = "mhlo.fusion"(%378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %380 = "mhlo.fusion"(%arg165, %arg166, %379, %374, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %381 = stablehlo.dot_general %380, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %382 = "mhlo.fusion"(%381, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %383 = stablehlo.dot_general %380, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %384 = "mhlo.fusion"(%383, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %385 = stablehlo.dot_general %382, %384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %386 = mhlo.bitcast %385 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %387 = "mhlo.fusion"(%386, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %388 = "mhlo.fusion"(%387, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %389 = "mhlo.fusion"(%388, %385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %390 = "mhlo.fusion"(%389, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %391 = "mhlo.fusion"(%385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %392 = "mhlo.fusion"(%391, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %393 = "mhlo.fusion"(%392, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %394 = "mhlo.fusion"(%389, %390, %393) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %395 = stablehlo.dot_general %380, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %396 = "mhlo.fusion"(%395, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %397 = stablehlo.dot_general %394, %396, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %398 = "mhlo.fusion"(%397) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %399 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %400 = stablehlo.dot_general %398, %399, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %401 = "mhlo.fusion"(%374, %400, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %402 = "mhlo.fusion"(%401, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %403 = "mhlo.fusion"(%401, %402) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %404 = "mhlo.fusion"(%403, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %405 = "mhlo.fusion"(%404, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %406 = "mhlo.fusion"(%405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %407 = "mhlo.fusion"(%arg175, %arg176, %406, %401, %402) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %408 = stablehlo.dot_general %407, %arg177, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %409 = "mhlo.fusion"(%408, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %410 = stablehlo.dot_general %409, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %411 = "mhlo.fusion"(%401, %410, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %412 = "mhlo.fusion"(%411, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %413 = "mhlo.fusion"(%411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %414 = "mhlo.fusion"(%413, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %415 = "mhlo.fusion"(%414, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %416 = "mhlo.fusion"(%415) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %417 = "mhlo.fusion"(%arg181, %arg182, %416, %411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %418 = stablehlo.dot_general %417, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %419 = "mhlo.fusion"(%418, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %463 = mhlo.copy %462 : tensor<8x12x197x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %464 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %420 = stablehlo.dot_general %417, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %421 = "mhlo.fusion"(%420, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x768xf32>
      %460 = stablehlo.multiply %458, %459 : tensor<1576x768xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,64,197]{2,1,3,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x64x197xf32>
      %463 = mhlo.copy %462 : tensor<8x12x64x197xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x64x197xf32>) -> tensor<96x64x197xf32>
      mhlo.return %464 : tensor<96x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x64x197xf32>
    %422 = stablehlo.dot_general %419, %421, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x64xf32>, tensor<96x64x197xf32>) -> tensor<96x197x197xf32>
    %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %424 = "mhlo.fusion"(%423, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
      mhlo.return %457 : tensor<8x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197x7xf32>
    %425 = "mhlo.fusion"(%424, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %426 = "mhlo.fusion"(%425, %422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197xf32>, %arg202: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197xf32>
      %459 = stablehlo.maximum %458, %arg201 : tensor<8x12x197xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.subtract %457, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.exponential %461 : tensor<8x12x197x197xf32>
      mhlo.return %462 : tensor<8x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197xf32>, tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
    %427 = "mhlo.fusion"(%426, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
      mhlo.return %457 : tensor<8x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<f32>) -> tensor<8x12x197xf32>
    %428 = "mhlo.fusion"(%422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x197xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %459 = stablehlo.compare EQ, %457, %458 : (tensor<8x12x197x197xf32>, tensor<8x12x197x197xf32>) -> tensor<8x12x197x197xi1>
      %460 = stablehlo.not %459 : tensor<8x12x197x197xi1>
      mhlo.return %460 : tensor<8x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<96x197x197xf32>) -> tensor<8x12x197x197xi1>
    %429 = "mhlo.fusion"(%428, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xi1>, %arg202: tensor<i1>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %458 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %458 : tensor<i1>
      }) : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
      mhlo.return %457 : tensor<8x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xi1>, tensor<i1>) -> tensor<8x12x197x7xi1>
    %430 = "mhlo.fusion"(%429, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x7xi1>, %arg202: tensor<i1>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
      mhlo.return %457 : tensor<8x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x7xi1>, tensor<i1>) -> tensor<8x12x197xi1>
    %431 = "mhlo.fusion"(%426, %427, %430) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x12x197x197xf32>, %arg202: tensor<8x12x197xf32>, %arg203: tensor<8x12x197xi1>):
      %457 = stablehlo.not %arg203 : tensor<8x12x197xi1>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<8x12x197xi1>) -> tensor<8x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x12x197x197xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<8x12x197xf32>) -> tensor<8x12x197x197xf32>
      %461 = stablehlo.divide %arg201, %460 : tensor<8x12x197x197xf32>
      %462 = stablehlo.select %458, %459, %461 : tensor<8x12x197x197xi1>, tensor<8x12x197x197xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x197xf32>) -> tensor<96x197x197xf32>
      mhlo.return %463 : tensor<96x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x12x197x197xf32>, tensor<8x12x197xf32>, tensor<8x12x197xi1>) -> tensor<96x197x197xf32>
    %432 = stablehlo.dot_general %417, %arg187, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<768x768xf32>) -> tensor<1576x768xf32>
    %433 = "mhlo.fusion"(%432, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x768xf32>, %arg202: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x12x64xf32>
      %460 = stablehlo.transpose %459, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,12,197,64]{3,1,2,0}"} : (tensor<8x197x12x64xf32>) -> tensor<8x12x197x64xf32>
      %461 = mhlo.copy %460 : tensor<8x12x197x64xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x12x197x64xf32>) -> tensor<96x197x64xf32>
      mhlo.return %462 : tensor<96x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x768xf32>, tensor<768xf32>) -> tensor<96x197x64xf32>
    %434 = stablehlo.dot_general %431, %433, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<96x197x197xf32>, tensor<96x197x64xf32>) -> tensor<96x197x64xf32>
    %435 = "mhlo.fusion"(%434) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x197x64xf32>):
      %457 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<96x197x64xf32>) -> tensor<8x12x197x64xf32>
      %458 = stablehlo.transpose %457, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,197,12,64]{3,1,2,0}"} : (tensor<8x12x197x64xf32>) -> tensor<8x197x12x64xf32>
      %459 = mhlo.copy %458 : tensor<8x197x12x64xf32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x197x12x64xf32>) -> tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<96x197x64xf32>) -> tensor<8x197x768xf32>
    %436 = "mhlo.fusion"(%arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %457 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %458 = mhlo.copy %457 : tensor<768x768xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
      mhlo.return %459 : tensor<8x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<8x768x768xf32>
    %437 = stablehlo.dot_general %435, %436, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x197x768xf32>, tensor<8x768x768xf32>) -> tensor<8x197x768xf32>
    %438 = "mhlo.fusion"(%411, %437, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %458 = stablehlo.add %arg202, %457 : tensor<8x197x768xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<8x197x768xf32>
      mhlo.return %459 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %439 = "mhlo.fusion"(%438, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %440 = "mhlo.fusion"(%438, %439) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %441 = "mhlo.fusion"(%440, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %442 = "mhlo.fusion"(%441, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %443 = "mhlo.fusion"(%442) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %444 = "mhlo.fusion"(%arg191, %arg192, %443, %438, %439) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x768xf32>) -> tensor<1576x768xf32>
      mhlo.return %467 : tensor<1576x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<1576x768xf32>
    %445 = stablehlo.dot_general %444, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x768xf32>, tensor<3072x768xf32>) -> tensor<1576x3072xf32>
    %446 = "mhlo.fusion"(%445, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1576x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<1576x3072xf32>
      %459 = stablehlo.add %458, %arg201 : tensor<1576x3072xf32>
      %460 = stablehlo.multiply %457, %459 : tensor<1576x3072xf32>
      %461 = mhlo.bitcast %460 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %462 = stablehlo.negate %459 : tensor<1576x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1576x3072xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<1576x3072xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x3072xf32>) -> tensor<8x197x3072xf32>
      %466 = stablehlo.abs %465 : tensor<8x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %468 = stablehlo.compare LT, %466, %467 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %470 = stablehlo.multiply %465, %465 : tensor<8x197x3072xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<8x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x197x3072xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<8x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x197x3072xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<8x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x197x3072xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<8x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x197x3072xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<8x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x197x3072xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<8x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x197x3072xf32>
      %489 = stablehlo.multiply %465, %488 : tensor<8x197x3072xf32>
      %490 = stablehlo.subtract %467, %489 : tensor<8x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %492 = stablehlo.compare LT, %465, %491 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %494 = stablehlo.negate %470 : tensor<8x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %496 = stablehlo.compare LT, %494, %495 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %497 = stablehlo.exponential %494 : tensor<8x197x3072xf32>
      %498 = stablehlo.divide %467, %466 : tensor<8x197x3072xf32>
      %499 = stablehlo.multiply %497, %498 : tensor<8x197x3072xf32>
      %500 = stablehlo.compare LT, %466, %493 : (tensor<8x197x3072xf32>, tensor<8x197x3072xf32>) -> tensor<8x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %502 = stablehlo.divide %467, %470 : tensor<8x197x3072xf32>
      %503 = stablehlo.multiply %501, %502 : tensor<8x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<8x197x3072xf32>
      %506 = stablehlo.multiply %505, %502 : tensor<8x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<8x197x3072xf32>
      %509 = stablehlo.multiply %508, %502 : tensor<8x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<8x197x3072xf32>
      %512 = stablehlo.multiply %511, %502 : tensor<8x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %513 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %514 = stablehlo.add %512, %513 : tensor<8x197x3072xf32>
      %515 = stablehlo.multiply %514, %502 : tensor<8x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %517 = stablehlo.add %515, %516 : tensor<8x197x3072xf32>
      %518 = stablehlo.multiply %517, %502 : tensor<8x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %520 = stablehlo.add %518, %519 : tensor<8x197x3072xf32>
      %521 = stablehlo.multiply %520, %502 : tensor<8x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %523 = stablehlo.add %521, %522 : tensor<8x197x3072xf32>
      %524 = stablehlo.multiply %523, %502 : tensor<8x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %526 = stablehlo.add %524, %525 : tensor<8x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %527 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %528 = stablehlo.multiply %527, %502 : tensor<8x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %530 = stablehlo.add %528, %529 : tensor<8x197x3072xf32>
      %531 = stablehlo.multiply %530, %502 : tensor<8x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %532 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %533 = stablehlo.add %531, %532 : tensor<8x197x3072xf32>
      %534 = stablehlo.multiply %533, %502 : tensor<8x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %536 = stablehlo.add %534, %535 : tensor<8x197x3072xf32>
      %537 = stablehlo.multiply %536, %502 : tensor<8x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %539 = stablehlo.add %537, %538 : tensor<8x197x3072xf32>
      %540 = stablehlo.multiply %539, %502 : tensor<8x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %542 = stablehlo.add %540, %541 : tensor<8x197x3072xf32>
      %543 = stablehlo.multiply %542, %502 : tensor<8x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %545 = stablehlo.add %543, %544 : tensor<8x197x3072xf32>
      %546 = stablehlo.multiply %545, %502 : tensor<8x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x197x3072xf32>
      %548 = stablehlo.add %546, %547 : tensor<8x197x3072xf32>
      %549 = stablehlo.select %500, %526, %548 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %550 = stablehlo.multiply %499, %549 : tensor<8x197x3072xf32>
      %551 = stablehlo.select %496, %491, %550 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %552 = stablehlo.subtract %493, %551 : tensor<8x197x3072xf32>
      %553 = stablehlo.select %492, %552, %551 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %554 = stablehlo.select %468, %490, %553 : tensor<8x197x3072xi1>, tensor<8x197x3072xf32>
      %555 = stablehlo.multiply %461, %554 : tensor<8x197x3072xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x197x3072xf32>) -> tensor<1576x3072xf32>
      mhlo.return %556 : tensor<1576x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1576x3072xf32>, tensor<3072xf32>) -> tensor<1576x3072xf32>
    %447 = stablehlo.dot_general %446, %arg195, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1576x3072xf32>, tensor<768x3072xf32>) -> tensor<1576x768xf32>
    %448 = "mhlo.fusion"(%438, %447, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<1576x768xf32>, %arg203: tensor<768xf32>):
      %457 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1576x768xf32>
      %458 = stablehlo.add %457, %arg202 : tensor<1576x768xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.add %459, %arg201 : tensor<8x197x768xf32>
      mhlo.return %460 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<1576x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    %449 = "mhlo.fusion"(%448, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197xf32>
    %450 = "mhlo.fusion"(%448, %449) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg202, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg201, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.multiply %460, %460 : tensor<8x197x768xf32>
      mhlo.return %461 : tensor<8x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x197x768xf32>
    %451 = "mhlo.fusion"(%450, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197x768xf32>, %arg202: tensor<f32>):
      %457 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %458 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %458 : tensor<f32>
      }) : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
      mhlo.return %457 : tensor<8x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x768xf32>, tensor<f32>) -> tensor<8x197x24xf32>
    %452 = "mhlo.fusion"(%451, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x197x24xf32>, %arg202: tensor<f32>):
      %457 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
      mhlo.return %457 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197x24xf32>, tensor<f32>) -> tensor<8x197xf32>
    %453 = "mhlo.fusion"(%452) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg201, %457 : tensor<8x197xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x197xf32>
      %461 = stablehlo.rsqrt %460 : tensor<8x197xf32>
      mhlo.return %461 : tensor<8x197xf32>
    }) {output_operand_aliasing = []} : (tensor<8x197xf32>) -> tensor<8x197xf32>
    %454 = "mhlo.fusion"(%arg197, %arg198, %453, %448, %449) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x197xf32>, %arg204: tensor<8x197x768xf32>, %arg205: tensor<8x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x197xf32>
      %458 = stablehlo.multiply %arg205, %457 : tensor<8x197xf32>
      %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %460 = stablehlo.subtract %arg204, %459 : tensor<8x197x768xf32>
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<8x197xf32>) -> tensor<8x197x768xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<8x197x768xf32>
      %463 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<8x197x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<8x197x768xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x197x768xf32>
      %467 = stablehlo.slice %466 [0:8, 0:1, 0:768] : (tensor<8x197x768xf32>) -> tensor<8x1x768xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x768xf32>) -> tensor<8x768xf32>
      mhlo.return %468 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x197xf32>, tensor<8x197x768xf32>, tensor<8x197xf32>) -> tensor<8x768xf32>
    %455 = stablehlo.dot_general %454, %arg199, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x768xf32>, tensor<1000x768xf32>) -> tensor<8x1000xf32>
    %456 = "mhlo.fusion"(%455, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1000xf32>, %arg202: tensor<1000xf32>):
      %457 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1000xf32>) -> tensor<8x1000xf32>
      %458 = stablehlo.add %457, %arg201 : tensor<8x1000xf32>
      mhlo.return %458 : tensor<8x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1000xf32>, tensor<1000xf32>) -> tensor<8x1000xf32>
    return %456 : tensor<8x1000xf32>
  }
}
