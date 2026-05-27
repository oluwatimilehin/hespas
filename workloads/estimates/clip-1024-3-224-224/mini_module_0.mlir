module {
  func.func @main(%arg0: tensor<1024x3x224x224xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<768xf32>, %arg3: tensor<50x768xf32>, %arg4: tensor<1x50xi32>, %arg5: tensor<768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768x768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768x768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<3072x768xf32>, %arg20: tensor<3072xf32>, %arg21: tensor<768x3072xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768x768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<768x768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<3072x768xf32>, %arg36: tensor<3072xf32>, %arg37: tensor<768x3072xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768x768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<3072x768xf32>, %arg52: tensor<3072xf32>, %arg53: tensor<768x3072xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768x768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<3072x768xf32>, %arg68: tensor<3072xf32>, %arg69: tensor<768x3072xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768x768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<3072x768xf32>, %arg84: tensor<3072xf32>, %arg85: tensor<768x3072xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<768x768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<3072x768xf32>, %arg100: tensor<3072xf32>, %arg101: tensor<768x3072xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768x768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<768x768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<3072x768xf32>, %arg116: tensor<3072xf32>, %arg117: tensor<768x3072xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768x768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<768x768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<3072x768xf32>, %arg132: tensor<3072xf32>, %arg133: tensor<768x3072xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768x768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<3072x768xf32>, %arg148: tensor<3072xf32>, %arg149: tensor<768x3072xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768x768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<768x768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<3072x768xf32>, %arg164: tensor<3072xf32>, %arg165: tensor<768x3072xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768x768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<768x768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<3072x768xf32>, %arg180: tensor<3072xf32>, %arg181: tensor<768x3072xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768x768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<3072x768xf32>, %arg196: tensor<3072xf32>, %arg197: tensor<768x3072xf32>, %arg198: tensor<768xf32>, %arg199: tensor<2x768xf32>, %arg200: tensor<2xf32>) -> tensor<1024x2xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x3x224x224xf32>):
      %461 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,224,224,3]{2,1,3,0}"} : (tensor<1024x3x224x224xf32>) -> tensor<1024x224x224x3xf32>
      %462 = mhlo.copy %461 : tensor<1024x224x224x3xf32>
      mhlo.return %462 : tensor<1024x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x3x224x224xf32>) -> tensor<1024x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x32x32xf32>):
      %461 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
      %462 = mhlo.copy %461 : tensor<32x32x3x768xf32>
      mhlo.return %462 : tensor<32x32x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [32, 32], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1024x224x224x3xf32>, tensor<32x32x3x768xf32>) -> tensor<1024x7x7x768xf32>
    %3 = "mhlo.fusion"(%2, %arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x7x7x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x1x768xf32>
      %462 = stablehlo.transpose %arg201, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,768,7,7]{1,3,2,0}"} : (tensor<1024x7x7x768xf32>) -> tensor<1024x768x7x7xf32>
      %463 = mhlo.copy %462 : tensor<1024x768x7x7xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x768x7x7xf32>) -> tensor<1024x768x49xf32>
      %465 = stablehlo.transpose %464, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1024,49,768]{1,2,0}"} : (tensor<1024x768x49xf32>) -> tensor<1024x49x768xf32>
      %466 = mhlo.copy %465 : tensor<1024x49x768xf32>
      %467 = stablehlo.concatenate %461, %466, dim = 1 : (tensor<1024x1x768xf32>, tensor<1024x49x768xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %467 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x7x7x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %4 = "mhlo.fusion"(%arg3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg202, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg202, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %467 = "stablehlo.gather"(%arg201, %466) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      mhlo.return %467 : tensor<50x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<50x1x768xf32>
    %5 = "mhlo.fusion"(%3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<50x1x768xf32>, %arg203: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg203, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg203, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg203 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg201, %476 : tensor<1024x50x768xf32>
      mhlo.return %477 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<1024x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %7 = "mhlo.fusion"(%6, %3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<50x1x768xf32>, %arg204: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg204, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg204, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg204 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg202, %476 : tensor<1024x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %479 = stablehlo.multiply %arg201, %478 : tensor<1024x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<1024x50x768xf32>
      %482 = stablehlo.multiply %481, %481 : tensor<1024x50x768xf32>
      mhlo.return %482 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<1024x50x768xf32>
    %8 = "mhlo.fusion"(%7, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %9 = "mhlo.fusion"(%8, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %10 = "mhlo.fusion"(%9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %11 = "mhlo.fusion"(%arg5, %arg6, %10, %6, %3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50xf32>, %arg205: tensor<1024x50x768xf32>, %arg206: tensor<50x1x768xf32>, %arg207: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg207, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg207, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg207 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg206 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg205, %476 : tensor<1024x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %479 = stablehlo.multiply %arg204, %478 : tensor<1024x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<1024x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<1024x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<1024x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<1024x50x768xf32>
      mhlo.return %487 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<1024x50x768xf32>
    %12 = "mhlo.fusion"(%11, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %13 = "mhlo.fusion"(%12, %arg5, %arg6, %10, %6, %3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<1024x50xf32>, %arg205: tensor<1024x50xf32>, %arg206: tensor<1024x50x768xf32>, %arg207: tensor<50x1x768xf32>, %arg208: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg208, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg208, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg208 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg206, %476 : tensor<1024x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %479 = stablehlo.multiply %arg205, %478 : tensor<1024x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<1024x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<1024x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<1024x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<1024x50x768xf32>
      %488 = stablehlo.multiply %arg201, %478 : tensor<1024x50xf32>
      %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %490 = stablehlo.subtract %487, %489 : tensor<1024x50x768xf32>
      %491 = stablehlo.multiply %490, %490 : tensor<1024x50x768xf32>
      mhlo.return %491 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<1024x50x768xf32>
    %14 = "mhlo.fusion"(%13, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %15 = "mhlo.fusion"(%14, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %16 = "mhlo.fusion"(%15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %17 = "mhlo.fusion"(%arg7, %arg8, %16, %12, %arg5, %arg6, %10, %6, %3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<1024x50xf32>, %arg208: tensor<1024x50xf32>, %arg209: tensor<1024x50x768xf32>, %arg210: tensor<50x1x768xf32>, %arg211: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg211, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg211, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg211 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg209, %476 : tensor<1024x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %479 = stablehlo.multiply %arg208, %478 : tensor<1024x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<1024x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg207, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<1024x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<1024x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<1024x50x768xf32>
      %488 = stablehlo.multiply %arg204, %478 : tensor<1024x50xf32>
      %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %490 = stablehlo.subtract %487, %489 : tensor<1024x50x768xf32>
      %491 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %492 = stablehlo.multiply %490, %491 : tensor<1024x50x768xf32>
      %493 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %494 = stablehlo.multiply %492, %493 : tensor<1024x50x768xf32>
      %495 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %496 = stablehlo.add %494, %495 : tensor<1024x50x768xf32>
      %497 = mhlo.bitcast %496 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %497 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<51200x768xf32>
    %18 = stablehlo.dot_general %17, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %19 = "mhlo.fusion"(%18, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %20 = stablehlo.dot_general %17, %arg11, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %21 = "mhlo.fusion"(%20, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %22 = stablehlo.dot_general %19, %21, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %24 = "mhlo.fusion"(%23, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %25 = "mhlo.fusion"(%24, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %26 = "mhlo.fusion"(%25, %22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %27 = "mhlo.fusion"(%26, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %28 = "mhlo.fusion"(%22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %29 = "mhlo.fusion"(%28, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %30 = "mhlo.fusion"(%29, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %31 = "mhlo.fusion"(%26, %27, %30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %32 = stablehlo.dot_general %17, %arg13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %33 = "mhlo.fusion"(%32, %arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %34 = stablehlo.dot_general %31, %33, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %35 = "mhlo.fusion"(%34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %36 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %37 = stablehlo.dot_general %35, %36, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %38 = "mhlo.fusion"(%37, %arg16, %arg5, %arg6, %10, %6, %3, %4, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1024x50xf32>, %arg206: tensor<1024x50xf32>, %arg207: tensor<1024x50x768xf32>, %arg208: tensor<50x1x768xf32>, %arg209: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg209, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg209, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg209 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<1024x50x768xf32>
      %477 = stablehlo.add %arg207, %476 : tensor<1024x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %479 = stablehlo.multiply %arg206, %478 : tensor<1024x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<1024x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<1024x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<1024x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<1024x50x768xf32>
      %488 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %489 = stablehlo.add %arg201, %488 : tensor<1024x50x768xf32>
      %490 = stablehlo.add %487, %489 : tensor<1024x50x768xf32>
      mhlo.return %490 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<1024x50x768xf32>
    %39 = "mhlo.fusion"(%38, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %40 = "mhlo.fusion"(%38, %39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %41 = "mhlo.fusion"(%40, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %42 = "mhlo.fusion"(%41, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %43 = "mhlo.fusion"(%42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %44 = "mhlo.fusion"(%arg17, %arg18, %43, %38, %39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %45 = stablehlo.dot_general %44, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %46 = "mhlo.fusion"(%45, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %47 = stablehlo.dot_general %46, %arg21, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %48 = "mhlo.fusion"(%38, %47, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %49 = "mhlo.fusion"(%48, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %50 = "mhlo.fusion"(%48, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %51 = "mhlo.fusion"(%50, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %52 = "mhlo.fusion"(%51, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %53 = "mhlo.fusion"(%52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %54 = "mhlo.fusion"(%arg23, %arg24, %53, %48, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %55 = stablehlo.dot_general %54, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %56 = "mhlo.fusion"(%55, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %57 = stablehlo.dot_general %54, %arg27, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %58 = "mhlo.fusion"(%57, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %59 = stablehlo.dot_general %56, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %60 = mhlo.bitcast %59 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %61 = "mhlo.fusion"(%60, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %62 = "mhlo.fusion"(%61, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %63 = "mhlo.fusion"(%62, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %64 = "mhlo.fusion"(%63, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %65 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %66 = "mhlo.fusion"(%65, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %67 = "mhlo.fusion"(%66, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %68 = "mhlo.fusion"(%63, %64, %67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %69 = stablehlo.dot_general %54, %arg29, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %70 = "mhlo.fusion"(%69, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %71 = stablehlo.dot_general %68, %70, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %73 = "mhlo.fusion"(%arg31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %74 = stablehlo.dot_general %72, %73, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %75 = "mhlo.fusion"(%48, %74, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %76 = "mhlo.fusion"(%75, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %77 = "mhlo.fusion"(%75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %78 = "mhlo.fusion"(%77, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %79 = "mhlo.fusion"(%78, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %81 = "mhlo.fusion"(%arg33, %arg34, %80, %75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %82 = stablehlo.dot_general %81, %arg35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %83 = "mhlo.fusion"(%82, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %84 = stablehlo.dot_general %83, %arg37, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %85 = "mhlo.fusion"(%75, %84, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %87 = "mhlo.fusion"(%85, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %88 = "mhlo.fusion"(%87, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %89 = "mhlo.fusion"(%88, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %91 = "mhlo.fusion"(%arg39, %arg40, %90, %85, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %92 = stablehlo.dot_general %91, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %93 = "mhlo.fusion"(%92, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %94 = stablehlo.dot_general %91, %arg43, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %95 = "mhlo.fusion"(%94, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %96 = stablehlo.dot_general %93, %95, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %98 = "mhlo.fusion"(%97, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %99 = "mhlo.fusion"(%98, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %100 = "mhlo.fusion"(%99, %96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %101 = "mhlo.fusion"(%100, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %102 = "mhlo.fusion"(%96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %103 = "mhlo.fusion"(%102, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %104 = "mhlo.fusion"(%103, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %105 = "mhlo.fusion"(%100, %101, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %106 = stablehlo.dot_general %91, %arg45, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %107 = "mhlo.fusion"(%106, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %108 = stablehlo.dot_general %105, %107, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %110 = "mhlo.fusion"(%arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %111 = stablehlo.dot_general %109, %110, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %112 = "mhlo.fusion"(%85, %111, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %113 = "mhlo.fusion"(%112, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %114 = "mhlo.fusion"(%112, %113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %115 = "mhlo.fusion"(%114, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %117 = "mhlo.fusion"(%116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %118 = "mhlo.fusion"(%arg49, %arg50, %117, %112, %113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %119 = stablehlo.dot_general %118, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %120 = "mhlo.fusion"(%119, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %121 = stablehlo.dot_general %120, %arg53, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %122 = "mhlo.fusion"(%112, %121, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %123 = "mhlo.fusion"(%122, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %124 = "mhlo.fusion"(%122, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %128 = "mhlo.fusion"(%arg55, %arg56, %127, %122, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %129 = stablehlo.dot_general %128, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %130 = "mhlo.fusion"(%129, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %131 = stablehlo.dot_general %128, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %132 = "mhlo.fusion"(%131, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %133 = stablehlo.dot_general %130, %132, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %135 = "mhlo.fusion"(%134, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %136 = "mhlo.fusion"(%135, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %137 = "mhlo.fusion"(%136, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %138 = "mhlo.fusion"(%137, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %139 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %140 = "mhlo.fusion"(%139, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %141 = "mhlo.fusion"(%140, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %142 = "mhlo.fusion"(%137, %138, %141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %143 = stablehlo.dot_general %128, %arg61, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %144 = "mhlo.fusion"(%143, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %145 = stablehlo.dot_general %142, %144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %146 = "mhlo.fusion"(%145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %147 = "mhlo.fusion"(%arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %148 = stablehlo.dot_general %146, %147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %149 = "mhlo.fusion"(%122, %148, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %150 = "mhlo.fusion"(%149, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %151 = "mhlo.fusion"(%149, %150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %152 = "mhlo.fusion"(%151, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %155 = "mhlo.fusion"(%arg65, %arg66, %154, %149, %150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %156 = stablehlo.dot_general %155, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %157 = "mhlo.fusion"(%156, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %158 = stablehlo.dot_general %157, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %159 = "mhlo.fusion"(%149, %158, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %160 = "mhlo.fusion"(%159, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %161 = "mhlo.fusion"(%159, %160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %162 = "mhlo.fusion"(%161, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %163 = "mhlo.fusion"(%162, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %165 = "mhlo.fusion"(%arg71, %arg72, %164, %159, %160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %166 = stablehlo.dot_general %165, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %167 = "mhlo.fusion"(%166, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %168 = stablehlo.dot_general %165, %arg75, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %169 = "mhlo.fusion"(%168, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %170 = stablehlo.dot_general %167, %169, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %172 = "mhlo.fusion"(%171, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %173 = "mhlo.fusion"(%172, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %174 = "mhlo.fusion"(%173, %170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %176 = "mhlo.fusion"(%170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %177 = "mhlo.fusion"(%176, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %178 = "mhlo.fusion"(%177, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %179 = "mhlo.fusion"(%174, %175, %178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %180 = stablehlo.dot_general %165, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %181 = "mhlo.fusion"(%180, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %182 = stablehlo.dot_general %179, %181, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %184 = "mhlo.fusion"(%arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %185 = stablehlo.dot_general %183, %184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %186 = "mhlo.fusion"(%159, %185, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %187 = "mhlo.fusion"(%186, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %188 = "mhlo.fusion"(%186, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %189 = "mhlo.fusion"(%188, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %190 = "mhlo.fusion"(%189, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %192 = "mhlo.fusion"(%arg81, %arg82, %191, %186, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %193 = stablehlo.dot_general %192, %arg83, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %194 = "mhlo.fusion"(%193, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %195 = stablehlo.dot_general %194, %arg85, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %196 = "mhlo.fusion"(%186, %195, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %197 = "mhlo.fusion"(%196, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %198 = "mhlo.fusion"(%196, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %200 = "mhlo.fusion"(%199, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %202 = "mhlo.fusion"(%arg87, %arg88, %201, %196, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %203 = stablehlo.dot_general %202, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %204 = "mhlo.fusion"(%203, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %205 = stablehlo.dot_general %202, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %206 = "mhlo.fusion"(%205, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %207 = stablehlo.dot_general %204, %206, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %209 = "mhlo.fusion"(%208, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %210 = "mhlo.fusion"(%209, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %211 = "mhlo.fusion"(%210, %207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %212 = "mhlo.fusion"(%211, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %213 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %214 = "mhlo.fusion"(%213, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %215 = "mhlo.fusion"(%214, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %216 = "mhlo.fusion"(%211, %212, %215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %217 = stablehlo.dot_general %202, %arg93, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %218 = "mhlo.fusion"(%217, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %219 = stablehlo.dot_general %216, %218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %221 = "mhlo.fusion"(%arg95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %222 = stablehlo.dot_general %220, %221, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %223 = "mhlo.fusion"(%196, %222, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %224 = "mhlo.fusion"(%223, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %225 = "mhlo.fusion"(%223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %226 = "mhlo.fusion"(%225, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %227 = "mhlo.fusion"(%226, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %229 = "mhlo.fusion"(%arg97, %arg98, %228, %223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %230 = stablehlo.dot_general %229, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %231 = "mhlo.fusion"(%230, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %232 = stablehlo.dot_general %231, %arg101, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %233 = "mhlo.fusion"(%223, %232, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %234 = "mhlo.fusion"(%233, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %235 = "mhlo.fusion"(%233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %236 = "mhlo.fusion"(%235, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %237 = "mhlo.fusion"(%236, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %238 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %239 = "mhlo.fusion"(%arg103, %arg104, %238, %233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %240 = stablehlo.dot_general %239, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %241 = "mhlo.fusion"(%240, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %242 = stablehlo.dot_general %239, %arg107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %243 = "mhlo.fusion"(%242, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %244 = stablehlo.dot_general %241, %243, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %246 = "mhlo.fusion"(%245, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %247 = "mhlo.fusion"(%246, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %248 = "mhlo.fusion"(%247, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %249 = "mhlo.fusion"(%248, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %250 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %251 = "mhlo.fusion"(%250, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %252 = "mhlo.fusion"(%251, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %253 = "mhlo.fusion"(%248, %249, %252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %254 = stablehlo.dot_general %239, %arg109, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %255 = "mhlo.fusion"(%254, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %256 = stablehlo.dot_general %253, %255, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %257 = "mhlo.fusion"(%256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %258 = "mhlo.fusion"(%arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %259 = stablehlo.dot_general %257, %258, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %260 = "mhlo.fusion"(%233, %259, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %261 = "mhlo.fusion"(%260, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %262 = "mhlo.fusion"(%260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %263 = "mhlo.fusion"(%262, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %264 = "mhlo.fusion"(%263, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %265 = "mhlo.fusion"(%264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %266 = "mhlo.fusion"(%arg113, %arg114, %265, %260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %267 = stablehlo.dot_general %266, %arg115, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %268 = "mhlo.fusion"(%267, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %269 = stablehlo.dot_general %268, %arg117, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %270 = "mhlo.fusion"(%260, %269, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %271 = "mhlo.fusion"(%270, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %272 = "mhlo.fusion"(%270, %271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %273 = "mhlo.fusion"(%272, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %276 = "mhlo.fusion"(%arg119, %arg120, %275, %270, %271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %277 = stablehlo.dot_general %276, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %278 = "mhlo.fusion"(%277, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %279 = stablehlo.dot_general %276, %arg123, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %280 = "mhlo.fusion"(%279, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %281 = stablehlo.dot_general %278, %280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %283 = "mhlo.fusion"(%282, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %284 = "mhlo.fusion"(%283, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %285 = "mhlo.fusion"(%284, %281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %286 = "mhlo.fusion"(%285, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %287 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %288 = "mhlo.fusion"(%287, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %289 = "mhlo.fusion"(%288, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %290 = "mhlo.fusion"(%285, %286, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %291 = stablehlo.dot_general %276, %arg125, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %292 = "mhlo.fusion"(%291, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %293 = stablehlo.dot_general %290, %292, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %294 = "mhlo.fusion"(%293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %295 = "mhlo.fusion"(%arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %296 = stablehlo.dot_general %294, %295, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %297 = "mhlo.fusion"(%270, %296, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %298 = "mhlo.fusion"(%297, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %299 = "mhlo.fusion"(%297, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %300 = "mhlo.fusion"(%299, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %302 = "mhlo.fusion"(%301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %303 = "mhlo.fusion"(%arg129, %arg130, %302, %297, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %304 = stablehlo.dot_general %303, %arg131, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %305 = "mhlo.fusion"(%304, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %306 = stablehlo.dot_general %305, %arg133, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %307 = "mhlo.fusion"(%297, %306, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %308 = "mhlo.fusion"(%307, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %309 = "mhlo.fusion"(%307, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %311 = "mhlo.fusion"(%310, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %312 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %313 = "mhlo.fusion"(%arg135, %arg136, %312, %307, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %314 = stablehlo.dot_general %313, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %315 = "mhlo.fusion"(%314, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %316 = stablehlo.dot_general %313, %arg139, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %317 = "mhlo.fusion"(%316, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %318 = stablehlo.dot_general %315, %317, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %320 = "mhlo.fusion"(%319, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %322 = "mhlo.fusion"(%321, %318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %323 = "mhlo.fusion"(%322, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %324 = "mhlo.fusion"(%318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %325 = "mhlo.fusion"(%324, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %326 = "mhlo.fusion"(%325, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %327 = "mhlo.fusion"(%322, %323, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %328 = stablehlo.dot_general %313, %arg141, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %329 = "mhlo.fusion"(%328, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %330 = stablehlo.dot_general %327, %329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %331 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %332 = "mhlo.fusion"(%arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %333 = stablehlo.dot_general %331, %332, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %334 = "mhlo.fusion"(%307, %333, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %335 = "mhlo.fusion"(%334, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %336 = "mhlo.fusion"(%334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %339 = "mhlo.fusion"(%338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %340 = "mhlo.fusion"(%arg145, %arg146, %339, %334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %341 = stablehlo.dot_general %340, %arg147, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %342 = "mhlo.fusion"(%341, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %343 = stablehlo.dot_general %342, %arg149, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %344 = "mhlo.fusion"(%334, %343, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %345 = "mhlo.fusion"(%344, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %346 = "mhlo.fusion"(%344, %345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %347 = "mhlo.fusion"(%346, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %348 = "mhlo.fusion"(%347, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %349 = "mhlo.fusion"(%348) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %350 = "mhlo.fusion"(%arg151, %arg152, %349, %344, %345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %351 = stablehlo.dot_general %350, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %352 = "mhlo.fusion"(%351, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %353 = stablehlo.dot_general %350, %arg155, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %354 = "mhlo.fusion"(%353, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %355 = stablehlo.dot_general %352, %354, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %357 = "mhlo.fusion"(%356, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %358 = "mhlo.fusion"(%357, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %359 = "mhlo.fusion"(%358, %355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %360 = "mhlo.fusion"(%359, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %361 = "mhlo.fusion"(%355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %362 = "mhlo.fusion"(%361, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %363 = "mhlo.fusion"(%362, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %364 = "mhlo.fusion"(%359, %360, %363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %365 = stablehlo.dot_general %350, %arg157, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %366 = "mhlo.fusion"(%365, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %367 = stablehlo.dot_general %364, %366, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %368 = "mhlo.fusion"(%367) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %369 = "mhlo.fusion"(%arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %370 = stablehlo.dot_general %368, %369, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %371 = "mhlo.fusion"(%344, %370, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %372 = "mhlo.fusion"(%371, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %373 = "mhlo.fusion"(%371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %374 = "mhlo.fusion"(%373, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %377 = "mhlo.fusion"(%arg161, %arg162, %376, %371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %378 = stablehlo.dot_general %377, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %379 = "mhlo.fusion"(%378, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %380 = stablehlo.dot_general %379, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %381 = "mhlo.fusion"(%371, %380, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %382 = "mhlo.fusion"(%381, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %383 = "mhlo.fusion"(%381, %382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %384 = "mhlo.fusion"(%383, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %385 = "mhlo.fusion"(%384, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %386 = "mhlo.fusion"(%385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %387 = "mhlo.fusion"(%arg167, %arg168, %386, %381, %382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %388 = stablehlo.dot_general %387, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %389 = "mhlo.fusion"(%388, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %390 = stablehlo.dot_general %387, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %391 = "mhlo.fusion"(%390, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %392 = stablehlo.dot_general %389, %391, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %394 = "mhlo.fusion"(%393, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %395 = "mhlo.fusion"(%394, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %396 = "mhlo.fusion"(%395, %392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %397 = "mhlo.fusion"(%396, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %398 = "mhlo.fusion"(%392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %399 = "mhlo.fusion"(%398, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %400 = "mhlo.fusion"(%399, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %401 = "mhlo.fusion"(%396, %397, %400) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %402 = stablehlo.dot_general %387, %arg173, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %403 = "mhlo.fusion"(%402, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %404 = stablehlo.dot_general %401, %403, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %405 = "mhlo.fusion"(%404) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %406 = "mhlo.fusion"(%arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %407 = stablehlo.dot_general %405, %406, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %408 = "mhlo.fusion"(%381, %407, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %409 = "mhlo.fusion"(%408, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %410 = "mhlo.fusion"(%408, %409) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %411 = "mhlo.fusion"(%410, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %412 = "mhlo.fusion"(%411, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %413 = "mhlo.fusion"(%412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %414 = "mhlo.fusion"(%arg177, %arg178, %413, %408, %409) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %415 = stablehlo.dot_general %414, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %416 = "mhlo.fusion"(%415, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %417 = stablehlo.dot_general %416, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %418 = "mhlo.fusion"(%408, %417, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %419 = "mhlo.fusion"(%418, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %420 = "mhlo.fusion"(%418, %419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %421 = "mhlo.fusion"(%420, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %422 = "mhlo.fusion"(%421, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %423 = "mhlo.fusion"(%422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %424 = "mhlo.fusion"(%arg183, %arg184, %423, %418, %419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %425 = stablehlo.dot_general %424, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %426 = "mhlo.fusion"(%425, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %468 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %427 = stablehlo.dot_general %424, %arg187, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %428 = "mhlo.fusion"(%427, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<51200x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<51200x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,64,50]{2,1,3,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<1024x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x64x50xf32>) -> tensor<12288x64x50xf32>
      mhlo.return %468 : tensor<12288x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x64x50xf32>
    %429 = stablehlo.dot_general %426, %428, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x64xf32>, tensor<12288x64x50xf32>) -> tensor<12288x50x50xf32>
    %430 = mhlo.bitcast %429 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %431 = "mhlo.fusion"(%430, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
      mhlo.return %461 : tensor<1024x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50x2xf32>
    %432 = "mhlo.fusion"(%431, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %433 = "mhlo.fusion"(%432, %429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50xf32>, %arg202: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<1024x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<1024x12x50x50xf32>
      mhlo.return %466 : tensor<1024x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50xf32>, tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
    %434 = "mhlo.fusion"(%433, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
      mhlo.return %461 : tensor<1024x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<f32>) -> tensor<1024x12x50xf32>
    %435 = "mhlo.fusion"(%429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<1024x12x50x50xf32>, tensor<1024x12x50x50xf32>) -> tensor<1024x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<1024x12x50x50xi1>
      mhlo.return %464 : tensor<1024x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12288x50x50xf32>) -> tensor<1024x12x50x50xi1>
    %436 = "mhlo.fusion"(%435, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
      mhlo.return %461 : tensor<1024x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xi1>, tensor<i1>) -> tensor<1024x12x50x2xi1>
    %437 = "mhlo.fusion"(%436, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
      mhlo.return %461 : tensor<1024x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x2xi1>, tensor<i1>) -> tensor<1024x12x50xi1>
    %438 = "mhlo.fusion"(%433, %434, %437) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x12x50x50xf32>, %arg202: tensor<1024x12x50xf32>, %arg203: tensor<1024x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<1024x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1024x12x50xi1>) -> tensor<1024x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<1024x12x50xf32>) -> tensor<1024x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<1024x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<1024x12x50x50xi1>, tensor<1024x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x50xf32>) -> tensor<12288x50x50xf32>
      mhlo.return %467 : tensor<12288x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x12x50x50xf32>, tensor<1024x12x50xf32>, tensor<1024x12x50xi1>) -> tensor<12288x50x50xf32>
    %439 = stablehlo.dot_general %424, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<768x768xf32>) -> tensor<51200x768xf32>
    %440 = "mhlo.fusion"(%439, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,12,50,64]{3,1,2,0}"} : (tensor<1024x50x12x64xf32>) -> tensor<1024x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<1024x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x12x50x64xf32>) -> tensor<12288x50x64xf32>
      mhlo.return %466 : tensor<12288x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x768xf32>, tensor<768xf32>) -> tensor<12288x50x64xf32>
    %441 = stablehlo.dot_general %438, %440, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12288x50x50xf32>, tensor<12288x50x64xf32>) -> tensor<12288x50x64xf32>
    %442 = "mhlo.fusion"(%441) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12288x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12288x50x64xf32>) -> tensor<1024x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1024,50,12,64]{3,1,2,0}"} : (tensor<1024x12x50x64xf32>) -> tensor<1024x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<1024x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x50x12x64xf32>) -> tensor<1024x50x768xf32>
      mhlo.return %464 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12288x50x64xf32>) -> tensor<1024x50x768xf32>
    %443 = "mhlo.fusion"(%arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
      mhlo.return %463 : tensor<1024x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<1024x768x768xf32>
    %444 = stablehlo.dot_general %442, %443, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x50x768xf32>, tensor<1024x768x768xf32>) -> tensor<1024x50x768xf32>
    %445 = "mhlo.fusion"(%418, %444, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<1024x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<1024x50x768xf32>
      mhlo.return %463 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50x768xf32>, tensor<768xf32>) -> tensor<1024x50x768xf32>
    %446 = "mhlo.fusion"(%445, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %447 = "mhlo.fusion"(%445, %446) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<1024x50x768xf32>
      mhlo.return %465 : tensor<1024x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
    %448 = "mhlo.fusion"(%447, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
      mhlo.return %461 : tensor<1024x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<f32>) -> tensor<1024x50x24xf32>
    %449 = "mhlo.fusion"(%448, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
      mhlo.return %461 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x24xf32>, tensor<f32>) -> tensor<1024x50xf32>
    %450 = "mhlo.fusion"(%449) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<1024x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<1024x50xf32>
      mhlo.return %465 : tensor<1024x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50xf32>) -> tensor<1024x50xf32>
    %451 = "mhlo.fusion"(%arg193, %arg194, %450, %445, %446) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1024x50xf32>, %arg204: tensor<1024x50x768xf32>, %arg205: tensor<1024x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<1024x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<1024x50xf32>) -> tensor<1024x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1024x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<1024x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1024x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<1024x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x768xf32>) -> tensor<51200x768xf32>
      mhlo.return %471 : tensor<51200x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1024x50xf32>, tensor<1024x50x768xf32>, tensor<1024x50xf32>) -> tensor<51200x768xf32>
    %452 = stablehlo.dot_general %451, %arg195, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x768xf32>, tensor<3072x768xf32>) -> tensor<51200x3072xf32>
    %453 = "mhlo.fusion"(%452, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<51200x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<51200x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<51200x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x3072xf32>) -> tensor<1024x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<1024x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<1024x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<1024x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<1024x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<1024x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<1024x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1024x50x3072xf32>) -> tensor<51200x3072xf32>
      mhlo.return %472 : tensor<51200x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<51200x3072xf32>, tensor<3072xf32>) -> tensor<51200x3072xf32>
    %454 = stablehlo.dot_general %453, %arg197, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<51200x3072xf32>, tensor<768x3072xf32>) -> tensor<51200x768xf32>
    %455 = "mhlo.fusion"(%445, %454, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x50x768xf32>, %arg202: tensor<51200x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<51200x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<51200x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<51200x768xf32>) -> tensor<1024x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<1024x50x768xf32>
      %465 = stablehlo.slice %464 [0:1024, 1:50, 0:768] : (tensor<1024x50x768xf32>) -> tensor<1024x49x768xf32>
      mhlo.return %465 : tensor<1024x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x50x768xf32>, tensor<51200x768xf32>, tensor<768xf32>) -> tensor<1024x49x768xf32>
    %456 = "mhlo.fusion"(%455, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x49x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<1024x49x768xf32>, tensor<f32>) -> tensor<1024x2x768xf32>
      mhlo.return %461 : tensor<1024x2x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x49x768xf32>, tensor<f32>) -> tensor<1024x2x768xf32>
    %457 = "mhlo.fusion"(%456, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1024x2x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1] : (tensor<1024x2x768xf32>, tensor<f32>) -> tensor<1024x768xf32>
      mhlo.return %461 : tensor<1024x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x2x768xf32>, tensor<f32>) -> tensor<1024x768xf32>
    %458 = "mhlo.fusion"(%457) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x768xf32>):
      %cst_1 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x768xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<1024x768xf32>
      mhlo.return %462 : tensor<1024x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x768xf32>) -> tensor<1024x768xf32>
    %459 = stablehlo.dot_general %458, %arg199, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x768xf32>, tensor<2x768xf32>) -> tensor<1024x2xf32>
    %460 = "mhlo.fusion"(%459, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1024x2xf32>, %arg202: tensor<2xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<2xf32>) -> tensor<1024x2xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<1024x2xf32>
      mhlo.return %462 : tensor<1024x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x2xf32>, tensor<2xf32>) -> tensor<1024x2xf32>
    return %460 : tensor<1024x2xf32>
  }
}
