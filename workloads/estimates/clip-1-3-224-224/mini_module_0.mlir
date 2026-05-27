module {
  func.func @main(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<1x50xi32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768x768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768x768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<3072x768xf32>, %arg20: tensor<3072xf32>, %arg21: tensor<768x3072xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768x768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<768x768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<3072x768xf32>, %arg36: tensor<3072xf32>, %arg37: tensor<768x3072xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768x768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768x768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<3072x768xf32>, %arg52: tensor<3072xf32>, %arg53: tensor<768x3072xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<768x768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<3072x768xf32>, %arg68: tensor<3072xf32>, %arg69: tensor<768x3072xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768x768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<768x768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<3072x768xf32>, %arg84: tensor<3072xf32>, %arg85: tensor<768x3072xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<768x768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<3072x768xf32>, %arg100: tensor<3072xf32>, %arg101: tensor<768x3072xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768x768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<768x768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<3072x768xf32>, %arg116: tensor<3072xf32>, %arg117: tensor<768x3072xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768x768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<768x768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<3072x768xf32>, %arg132: tensor<3072xf32>, %arg133: tensor<768x3072xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768x768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<3072x768xf32>, %arg148: tensor<3072xf32>, %arg149: tensor<768x3072xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768x768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<768x768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<3072x768xf32>, %arg164: tensor<3072xf32>, %arg165: tensor<768x3072xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768x768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<768x768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<3072x768xf32>, %arg180: tensor<3072xf32>, %arg181: tensor<768x3072xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768x768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768x768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<3072x768xf32>, %arg196: tensor<3072xf32>, %arg197: tensor<768x3072xf32>, %arg198: tensor<768xf32>, %arg199: tensor<2x768xf32>, %arg200: tensor<2xf32>) -> tensor<1x2xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x3x224x224xf32>):
      %434 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %435 = mhlo.copy %434 : tensor<1x224x224x3xf32>
      mhlo.return %435 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x32x32xf32>):
      %434 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
      %435 = mhlo.copy %434 : tensor<32x32x3x768xf32>
      mhlo.return %435 : tensor<32x32x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [32, 32], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<32x32x3x768xf32>) -> tensor<1x7x7x768xf32>
    %3 = "mhlo.fusion"(%arg2, %arg3, %2, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>, %arg203: tensor<1x7x7x768xf32>, %arg204: tensor<768xf32>):
      %434 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x1x768xf32>
      %435 = stablehlo.transpose %arg203, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,7,7]{1,3,2,0}"} : (tensor<1x7x7x768xf32>) -> tensor<1x768x7x7xf32>
      %436 = mhlo.copy %435 : tensor<1x768x7x7xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<1x768x49xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,49,768]{1,2,0}"} : (tensor<1x768x49xf32>) -> tensor<1x49x768xf32>
      %439 = mhlo.copy %438 : tensor<1x49x768xf32>
      %440 = stablehlo.concatenate %434, %439, dim = 1 : (tensor<1x1x768xf32>, tensor<1x49x768xf32>) -> tensor<1x50x768xf32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LT, %arg202, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %443 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %444 = stablehlo.add %arg202, %443 : tensor<1x50xi32>
      %445 = stablehlo.select %442, %444, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %446 = stablehlo.compare GE, %445, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %447 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %448 = stablehlo.compare LE, %445, %447 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %449 = stablehlo.and %446, %448 : tensor<1x50xi1>
      %450 = mhlo.bitcast %449 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %451 = stablehlo.broadcast_in_dim %450, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %452 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %453 = "stablehlo.gather"(%arg201, %452) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      %454 = mhlo.bitcast %453 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %456 = stablehlo.select %451, %454, %455 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %457 = stablehlo.add %440, %456 : tensor<1x50x768xf32>
      mhlo.return %457 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %5 = "mhlo.fusion"(%3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %7 = "mhlo.fusion"(%6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %8 = "mhlo.fusion"(%arg5, %arg6, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      mhlo.return %445 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %9 = "mhlo.fusion"(%8, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %10 = "mhlo.fusion"(%9, %arg5, %arg6, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<1x50xf32>, %arg205: tensor<1x50x768xf32>, %arg206: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg206, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg205, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = stablehlo.multiply %arg201, %434 : tensor<1x50xf32>
      %447 = mhlo.bitcast %446 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %448 = stablehlo.broadcast_in_dim %447, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %449 = stablehlo.subtract %445, %448 : tensor<1x50x768xf32>
      %450 = stablehlo.multiply %449, %449 : tensor<1x50x768xf32>
      mhlo.return %450 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %11 = "mhlo.fusion"(%10, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %12 = "mhlo.fusion"(%11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %13 = "mhlo.fusion"(%arg7, %arg8, %12, %9, %arg5, %arg6, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<1x50xf32>, %arg208: tensor<1x50x768xf32>, %arg209: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg209, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg208, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg207 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = stablehlo.multiply %arg204, %434 : tensor<1x50xf32>
      %447 = mhlo.bitcast %446 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %448 = stablehlo.broadcast_in_dim %447, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %449 = stablehlo.subtract %445, %448 : tensor<1x50x768xf32>
      %450 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %451 = stablehlo.broadcast_in_dim %450, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %452 = stablehlo.multiply %449, %451 : tensor<1x50x768xf32>
      %453 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %454 = stablehlo.multiply %452, %453 : tensor<1x50x768xf32>
      %455 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x50x768xf32>
      %457 = mhlo.bitcast %456 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %457 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %14 = stablehlo.dot_general %13, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %15 = "mhlo.fusion"(%14, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %16 = stablehlo.dot_general %13, %arg11, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %17 = "mhlo.fusion"(%16, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %18 = stablehlo.dot_general %15, %17, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %20 = "mhlo.fusion"(%19, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %21 = "mhlo.fusion"(%20, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %22 = "mhlo.fusion"(%21, %18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %23 = "mhlo.fusion"(%22, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %24 = "mhlo.fusion"(%23, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %25 = "mhlo.fusion"(%18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %26 = "mhlo.fusion"(%25, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %27 = "mhlo.fusion"(%26, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %28 = "mhlo.fusion"(%22, %24, %27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %29 = stablehlo.dot_general %13, %arg13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %30 = "mhlo.fusion"(%29, %arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %31 = stablehlo.dot_general %28, %30, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %32 = "mhlo.fusion"(%31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %33 = stablehlo.dot_general %32, %arg15, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %34 = "mhlo.fusion"(%33, %arg16, %arg5, %arg6, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<1x50xf32>, %arg206: tensor<1x50x768xf32>, %arg207: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg207, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg206, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %447 = stablehlo.add %446, %arg201 : tensor<50x768xf32>
      %448 = mhlo.bitcast %447 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %449 = stablehlo.add %445, %448 : tensor<1x50x768xf32>
      mhlo.return %449 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %36 = "mhlo.fusion"(%34, %35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %37 = "mhlo.fusion"(%36, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %38 = "mhlo.fusion"(%37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %39 = "mhlo.fusion"(%arg17, %arg18, %38, %34, %35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %40 = stablehlo.dot_general %39, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %41 = "mhlo.fusion"(%40, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %42 = stablehlo.dot_general %41, %arg21, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %43 = "mhlo.fusion"(%34, %42, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %44 = "mhlo.fusion"(%43, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %45 = "mhlo.fusion"(%43, %44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %46 = "mhlo.fusion"(%45, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %47 = "mhlo.fusion"(%46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %48 = "mhlo.fusion"(%arg23, %arg24, %47, %43, %44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %49 = stablehlo.dot_general %48, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %50 = "mhlo.fusion"(%49, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %51 = stablehlo.dot_general %48, %arg27, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %52 = "mhlo.fusion"(%51, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %53 = stablehlo.dot_general %50, %52, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %55 = "mhlo.fusion"(%54, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %56 = "mhlo.fusion"(%55, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %57 = "mhlo.fusion"(%56, %53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %58 = "mhlo.fusion"(%57, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %59 = "mhlo.fusion"(%58, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %60 = "mhlo.fusion"(%53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %61 = "mhlo.fusion"(%60, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %62 = "mhlo.fusion"(%61, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %63 = "mhlo.fusion"(%57, %59, %62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %64 = stablehlo.dot_general %48, %arg29, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %65 = "mhlo.fusion"(%64, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %66 = stablehlo.dot_general %63, %65, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %67 = "mhlo.fusion"(%66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %68 = stablehlo.dot_general %67, %arg31, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %69 = "mhlo.fusion"(%43, %68, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %70 = "mhlo.fusion"(%69, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %71 = "mhlo.fusion"(%69, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %72 = "mhlo.fusion"(%71, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %74 = "mhlo.fusion"(%arg33, %arg34, %73, %69, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %75 = stablehlo.dot_general %74, %arg35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %76 = "mhlo.fusion"(%75, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %77 = stablehlo.dot_general %76, %arg37, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %78 = "mhlo.fusion"(%69, %77, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %79 = "mhlo.fusion"(%78, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %80 = "mhlo.fusion"(%78, %79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %81 = "mhlo.fusion"(%80, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %82 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %83 = "mhlo.fusion"(%arg39, %arg40, %82, %78, %79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %84 = stablehlo.dot_general %83, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %85 = "mhlo.fusion"(%84, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %86 = stablehlo.dot_general %83, %arg43, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %87 = "mhlo.fusion"(%86, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %88 = stablehlo.dot_general %85, %87, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %90 = "mhlo.fusion"(%89, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %91 = "mhlo.fusion"(%90, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %92 = "mhlo.fusion"(%91, %88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %93 = "mhlo.fusion"(%92, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %94 = "mhlo.fusion"(%93, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %95 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %96 = "mhlo.fusion"(%95, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %97 = "mhlo.fusion"(%96, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %98 = "mhlo.fusion"(%92, %94, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %99 = stablehlo.dot_general %83, %arg45, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %100 = "mhlo.fusion"(%99, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %101 = stablehlo.dot_general %98, %100, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %102 = "mhlo.fusion"(%101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %103 = stablehlo.dot_general %102, %arg47, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %104 = "mhlo.fusion"(%78, %103, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %105 = "mhlo.fusion"(%104, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %106 = "mhlo.fusion"(%104, %105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %108 = "mhlo.fusion"(%107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %109 = "mhlo.fusion"(%arg49, %arg50, %108, %104, %105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %110 = stablehlo.dot_general %109, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %111 = "mhlo.fusion"(%110, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %112 = stablehlo.dot_general %111, %arg53, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %113 = "mhlo.fusion"(%104, %112, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %114 = "mhlo.fusion"(%113, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %115 = "mhlo.fusion"(%113, %114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %117 = "mhlo.fusion"(%116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %118 = "mhlo.fusion"(%arg55, %arg56, %117, %113, %114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %119 = stablehlo.dot_general %118, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %120 = "mhlo.fusion"(%119, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %121 = stablehlo.dot_general %118, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %122 = "mhlo.fusion"(%121, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %123 = stablehlo.dot_general %120, %122, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %124 = mhlo.bitcast %123 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %125 = "mhlo.fusion"(%124, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %126 = "mhlo.fusion"(%125, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %127 = "mhlo.fusion"(%126, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %128 = "mhlo.fusion"(%127, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %129 = "mhlo.fusion"(%128, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %130 = "mhlo.fusion"(%123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %131 = "mhlo.fusion"(%130, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %132 = "mhlo.fusion"(%131, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %133 = "mhlo.fusion"(%127, %129, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %134 = stablehlo.dot_general %118, %arg61, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %135 = "mhlo.fusion"(%134, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %136 = stablehlo.dot_general %133, %135, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %137 = "mhlo.fusion"(%136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %138 = stablehlo.dot_general %137, %arg63, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %139 = "mhlo.fusion"(%113, %138, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %140 = "mhlo.fusion"(%139, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %141 = "mhlo.fusion"(%139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %144 = "mhlo.fusion"(%arg65, %arg66, %143, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %145 = stablehlo.dot_general %144, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %146 = "mhlo.fusion"(%145, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %147 = stablehlo.dot_general %146, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %148 = "mhlo.fusion"(%139, %147, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %149 = "mhlo.fusion"(%148, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %150 = "mhlo.fusion"(%148, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %151 = "mhlo.fusion"(%150, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %152 = "mhlo.fusion"(%151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %153 = "mhlo.fusion"(%arg71, %arg72, %152, %148, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %154 = stablehlo.dot_general %153, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %155 = "mhlo.fusion"(%154, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %156 = stablehlo.dot_general %153, %arg75, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %157 = "mhlo.fusion"(%156, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %158 = stablehlo.dot_general %155, %157, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %159 = mhlo.bitcast %158 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %160 = "mhlo.fusion"(%159, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %161 = "mhlo.fusion"(%160, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %162 = "mhlo.fusion"(%161, %158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %163 = "mhlo.fusion"(%162, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %164 = "mhlo.fusion"(%163, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %165 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %166 = "mhlo.fusion"(%165, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %167 = "mhlo.fusion"(%166, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %168 = "mhlo.fusion"(%162, %164, %167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %169 = stablehlo.dot_general %153, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %170 = "mhlo.fusion"(%169, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %171 = stablehlo.dot_general %168, %170, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %173 = stablehlo.dot_general %172, %arg79, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %174 = "mhlo.fusion"(%148, %173, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %176 = "mhlo.fusion"(%174, %175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %177 = "mhlo.fusion"(%176, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %178 = "mhlo.fusion"(%177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %179 = "mhlo.fusion"(%arg81, %arg82, %178, %174, %175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %180 = stablehlo.dot_general %179, %arg83, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %181 = "mhlo.fusion"(%180, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %182 = stablehlo.dot_general %181, %arg85, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %183 = "mhlo.fusion"(%174, %182, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %185 = "mhlo.fusion"(%183, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %186 = "mhlo.fusion"(%185, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %188 = "mhlo.fusion"(%arg87, %arg88, %187, %183, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %189 = stablehlo.dot_general %188, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %190 = "mhlo.fusion"(%189, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %191 = stablehlo.dot_general %188, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %192 = "mhlo.fusion"(%191, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %193 = stablehlo.dot_general %190, %192, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %194 = mhlo.bitcast %193 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %195 = "mhlo.fusion"(%194, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %196 = "mhlo.fusion"(%195, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %197 = "mhlo.fusion"(%196, %193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %198 = "mhlo.fusion"(%197, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %200 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %201 = "mhlo.fusion"(%200, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %202 = "mhlo.fusion"(%201, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %203 = "mhlo.fusion"(%197, %199, %202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %204 = stablehlo.dot_general %188, %arg93, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %205 = "mhlo.fusion"(%204, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %206 = stablehlo.dot_general %203, %205, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %207 = "mhlo.fusion"(%206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %208 = stablehlo.dot_general %207, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %209 = "mhlo.fusion"(%183, %208, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %211 = "mhlo.fusion"(%209, %210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %212 = "mhlo.fusion"(%211, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %213 = "mhlo.fusion"(%212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %214 = "mhlo.fusion"(%arg97, %arg98, %213, %209, %210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %215 = stablehlo.dot_general %214, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %216 = "mhlo.fusion"(%215, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %217 = stablehlo.dot_general %216, %arg101, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %218 = "mhlo.fusion"(%209, %217, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %219 = "mhlo.fusion"(%218, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %220 = "mhlo.fusion"(%218, %219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %221 = "mhlo.fusion"(%220, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %222 = "mhlo.fusion"(%221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %223 = "mhlo.fusion"(%arg103, %arg104, %222, %218, %219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %224 = stablehlo.dot_general %223, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %225 = "mhlo.fusion"(%224, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %226 = stablehlo.dot_general %223, %arg107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %227 = "mhlo.fusion"(%226, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %228 = stablehlo.dot_general %225, %227, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %230 = "mhlo.fusion"(%229, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %231 = "mhlo.fusion"(%230, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %232 = "mhlo.fusion"(%231, %228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %233 = "mhlo.fusion"(%232, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %234 = "mhlo.fusion"(%233, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %235 = "mhlo.fusion"(%228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %236 = "mhlo.fusion"(%235, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %237 = "mhlo.fusion"(%236, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %238 = "mhlo.fusion"(%232, %234, %237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %239 = stablehlo.dot_general %223, %arg109, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %240 = "mhlo.fusion"(%239, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %241 = stablehlo.dot_general %238, %240, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %242 = "mhlo.fusion"(%241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %243 = stablehlo.dot_general %242, %arg111, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %244 = "mhlo.fusion"(%218, %243, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %245 = "mhlo.fusion"(%244, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %246 = "mhlo.fusion"(%244, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %247 = "mhlo.fusion"(%246, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %248 = "mhlo.fusion"(%247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %249 = "mhlo.fusion"(%arg113, %arg114, %248, %244, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %250 = stablehlo.dot_general %249, %arg115, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %251 = "mhlo.fusion"(%250, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %252 = stablehlo.dot_general %251, %arg117, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %253 = "mhlo.fusion"(%244, %252, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %254 = "mhlo.fusion"(%253, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %255 = "mhlo.fusion"(%253, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %257 = "mhlo.fusion"(%256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %258 = "mhlo.fusion"(%arg119, %arg120, %257, %253, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %259 = stablehlo.dot_general %258, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %260 = "mhlo.fusion"(%259, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %261 = stablehlo.dot_general %258, %arg123, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %262 = "mhlo.fusion"(%261, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %263 = stablehlo.dot_general %260, %262, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %265 = "mhlo.fusion"(%264, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %266 = "mhlo.fusion"(%265, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %267 = "mhlo.fusion"(%266, %263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %268 = "mhlo.fusion"(%267, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %269 = "mhlo.fusion"(%268, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %270 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %271 = "mhlo.fusion"(%270, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %272 = "mhlo.fusion"(%271, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %273 = "mhlo.fusion"(%267, %269, %272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %274 = stablehlo.dot_general %258, %arg125, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %275 = "mhlo.fusion"(%274, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %276 = stablehlo.dot_general %273, %275, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %278 = stablehlo.dot_general %277, %arg127, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %279 = "mhlo.fusion"(%253, %278, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %280 = "mhlo.fusion"(%279, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %281 = "mhlo.fusion"(%279, %280) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %282 = "mhlo.fusion"(%281, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %283 = "mhlo.fusion"(%282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %284 = "mhlo.fusion"(%arg129, %arg130, %283, %279, %280) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %285 = stablehlo.dot_general %284, %arg131, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %286 = "mhlo.fusion"(%285, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %287 = stablehlo.dot_general %286, %arg133, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %288 = "mhlo.fusion"(%279, %287, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %289 = "mhlo.fusion"(%288, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %290 = "mhlo.fusion"(%288, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %291 = "mhlo.fusion"(%290, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %292 = "mhlo.fusion"(%291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %293 = "mhlo.fusion"(%arg135, %arg136, %292, %288, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %294 = stablehlo.dot_general %293, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %295 = "mhlo.fusion"(%294, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %296 = stablehlo.dot_general %293, %arg139, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %297 = "mhlo.fusion"(%296, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %298 = stablehlo.dot_general %295, %297, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %300 = "mhlo.fusion"(%299, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %301 = "mhlo.fusion"(%300, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %302 = "mhlo.fusion"(%301, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %304 = "mhlo.fusion"(%303, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %305 = "mhlo.fusion"(%298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %306 = "mhlo.fusion"(%305, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %307 = "mhlo.fusion"(%306, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %308 = "mhlo.fusion"(%302, %304, %307) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %309 = stablehlo.dot_general %293, %arg141, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %310 = "mhlo.fusion"(%309, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %311 = stablehlo.dot_general %308, %310, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %312 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %313 = stablehlo.dot_general %312, %arg143, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %314 = "mhlo.fusion"(%288, %313, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %315 = "mhlo.fusion"(%314, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %316 = "mhlo.fusion"(%314, %315) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %317 = "mhlo.fusion"(%316, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %318 = "mhlo.fusion"(%317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %319 = "mhlo.fusion"(%arg145, %arg146, %318, %314, %315) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %320 = stablehlo.dot_general %319, %arg147, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %321 = "mhlo.fusion"(%320, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %322 = stablehlo.dot_general %321, %arg149, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %323 = "mhlo.fusion"(%314, %322, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %324 = "mhlo.fusion"(%323, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %325 = "mhlo.fusion"(%323, %324) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %326 = "mhlo.fusion"(%325, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %327 = "mhlo.fusion"(%326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %328 = "mhlo.fusion"(%arg151, %arg152, %327, %323, %324) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %329 = stablehlo.dot_general %328, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %330 = "mhlo.fusion"(%329, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %331 = stablehlo.dot_general %328, %arg155, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %332 = "mhlo.fusion"(%331, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %333 = stablehlo.dot_general %330, %332, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %334 = mhlo.bitcast %333 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %335 = "mhlo.fusion"(%334, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %336 = "mhlo.fusion"(%335, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %337 = "mhlo.fusion"(%336, %333) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %339 = "mhlo.fusion"(%338, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %340 = "mhlo.fusion"(%333) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %341 = "mhlo.fusion"(%340, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %342 = "mhlo.fusion"(%341, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %343 = "mhlo.fusion"(%337, %339, %342) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %344 = stablehlo.dot_general %328, %arg157, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %345 = "mhlo.fusion"(%344, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %346 = stablehlo.dot_general %343, %345, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %348 = stablehlo.dot_general %347, %arg159, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %349 = "mhlo.fusion"(%323, %348, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %350 = "mhlo.fusion"(%349, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %351 = "mhlo.fusion"(%349, %350) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %352 = "mhlo.fusion"(%351, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %353 = "mhlo.fusion"(%352) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %354 = "mhlo.fusion"(%arg161, %arg162, %353, %349, %350) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %355 = stablehlo.dot_general %354, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %356 = "mhlo.fusion"(%355, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %357 = stablehlo.dot_general %356, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %358 = "mhlo.fusion"(%349, %357, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %359 = "mhlo.fusion"(%358, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %360 = "mhlo.fusion"(%358, %359) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %361 = "mhlo.fusion"(%360, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %362 = "mhlo.fusion"(%361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %363 = "mhlo.fusion"(%arg167, %arg168, %362, %358, %359) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %364 = stablehlo.dot_general %363, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %365 = "mhlo.fusion"(%364, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %366 = stablehlo.dot_general %363, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %367 = "mhlo.fusion"(%366, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %368 = stablehlo.dot_general %365, %367, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %369 = mhlo.bitcast %368 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %370 = "mhlo.fusion"(%369, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %371 = "mhlo.fusion"(%370, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %372 = "mhlo.fusion"(%371, %368) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %373 = "mhlo.fusion"(%372, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %374 = "mhlo.fusion"(%373, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %375 = "mhlo.fusion"(%368) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %376 = "mhlo.fusion"(%375, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %377 = "mhlo.fusion"(%376, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %378 = "mhlo.fusion"(%372, %374, %377) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %379 = stablehlo.dot_general %363, %arg173, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %380 = "mhlo.fusion"(%379, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %381 = stablehlo.dot_general %378, %380, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %382 = "mhlo.fusion"(%381) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %383 = stablehlo.dot_general %382, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %384 = "mhlo.fusion"(%358, %383, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %385 = "mhlo.fusion"(%384, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %386 = "mhlo.fusion"(%384, %385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %387 = "mhlo.fusion"(%386, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %388 = "mhlo.fusion"(%387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %389 = "mhlo.fusion"(%arg177, %arg178, %388, %384, %385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %390 = stablehlo.dot_general %389, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %391 = "mhlo.fusion"(%390, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %392 = stablehlo.dot_general %391, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %393 = "mhlo.fusion"(%384, %392, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %394 = "mhlo.fusion"(%393, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %395 = "mhlo.fusion"(%393, %394) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %396 = "mhlo.fusion"(%395, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %397 = "mhlo.fusion"(%396) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %398 = "mhlo.fusion"(%arg183, %arg184, %397, %393, %394) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %399 = stablehlo.dot_general %398, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %400 = "mhlo.fusion"(%399, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %440 = mhlo.copy %439 : tensor<1x12x50x64xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %441 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %401 = stablehlo.dot_general %398, %arg187, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %402 = "mhlo.fusion"(%401, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<50x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<50x768xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,50]{2,1,3,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x64x50xf32>
      %440 = mhlo.copy %439 : tensor<1x12x64x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x50xf32>) -> tensor<12x64x50xf32>
      mhlo.return %441 : tensor<12x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x64x50xf32>
    %403 = stablehlo.dot_general %400, %402, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %404 = mhlo.bitcast %403 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %405 = "mhlo.fusion"(%404, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %406 = "mhlo.fusion"(%405, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %407 = "mhlo.fusion"(%406, %403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50xf32>, %arg202: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50xf32>
      %436 = stablehlo.maximum %435, %arg201 : tensor<1x12x50xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %439 = stablehlo.subtract %434, %438 : tensor<1x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<1x12x50x50xf32>
      mhlo.return %440 : tensor<1x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50xf32>, tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
    %408 = "mhlo.fusion"(%407, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
      mhlo.return %434 : tensor<1x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<f32>) -> tensor<1x12x50x2xf32>
    %409 = "mhlo.fusion"(%408, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
      mhlo.return %434 : tensor<1x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xf32>, tensor<f32>) -> tensor<1x12x50xf32>
    %410 = "mhlo.fusion"(%403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %436 = stablehlo.compare EQ, %434, %435 : (tensor<1x12x50x50xf32>, tensor<1x12x50x50xf32>) -> tensor<1x12x50x50xi1>
      %437 = stablehlo.not %436 : tensor<1x12x50x50xi1>
      mhlo.return %437 : tensor<1x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<1x12x50x50xi1>
    %411 = "mhlo.fusion"(%410, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xi1>, %arg202: tensor<i1>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %435 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %435 : tensor<i1>
      }) : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
      mhlo.return %434 : tensor<1x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xi1>, tensor<i1>) -> tensor<1x12x50x2xi1>
    %412 = "mhlo.fusion"(%411, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x2xi1>, %arg202: tensor<i1>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
      mhlo.return %434 : tensor<1x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x2xi1>, tensor<i1>) -> tensor<1x12x50xi1>
    %413 = "mhlo.fusion"(%407, %409, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x50x50xf32>, %arg202: tensor<1x12x50xf32>, %arg203: tensor<1x12x50xi1>):
      %434 = stablehlo.not %arg203 : tensor<1x12x50xi1>
      %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xi1>) -> tensor<12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1, 2] : (tensor<12x50xi1>) -> tensor<1x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x50x50xf32>
      %438 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x50xf32>) -> tensor<12x50xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [1, 2] : (tensor<12x50xf32>) -> tensor<1x12x50x50xf32>
      %440 = stablehlo.divide %arg201, %439 : tensor<1x12x50x50xf32>
      %441 = stablehlo.select %436, %437, %440 : tensor<1x12x50x50xi1>, tensor<1x12x50x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %442 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x50x50xf32>, tensor<1x12x50xf32>, tensor<1x12x50xi1>) -> tensor<12x50x50xf32>
    %414 = stablehlo.dot_general %398, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %415 = "mhlo.fusion"(%414, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x12x64xf32>
      %437 = stablehlo.transpose %436, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,50,64]{3,1,2,0}"} : (tensor<1x50x12x64xf32>) -> tensor<1x12x50x64xf32>
      %438 = mhlo.copy %437 : tensor<1x12x50x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x50x64xf32>) -> tensor<12x50x64xf32>
      mhlo.return %439 : tensor<12x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>) -> tensor<12x50x64xf32>
    %416 = stablehlo.dot_general %413, %415, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x50xf32>, tensor<12x50x64xf32>) -> tensor<12x50x64xf32>
    %417 = "mhlo.fusion"(%416) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x50x64xf32>):
      %434 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x50x64xf32>) -> tensor<1x12x50x64xf32>
      %435 = stablehlo.transpose %434, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,50,12,64]{3,1,2,0}"} : (tensor<1x12x50x64xf32>) -> tensor<1x50x12x64xf32>
      %436 = mhlo.copy %435 : tensor<1x50x12x64xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x50x12x64xf32>) -> tensor<50x768xf32>
      mhlo.return %437 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x64xf32>) -> tensor<50x768xf32>
    %418 = stablehlo.dot_general %417, %arg191, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<50x768xf32>
    %419 = "mhlo.fusion"(%393, %418, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      mhlo.return %437 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x50x768xf32>
    %420 = "mhlo.fusion"(%419, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
      mhlo.return %434 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50xf32>
    %421 = "mhlo.fusion"(%419, %420) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg202, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<1x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<1x50x768xf32>
      mhlo.return %439 : tensor<1x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<1x50x768xf32>
    %422 = "mhlo.fusion"(%421, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
      mhlo.return %434 : tensor<1x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<f32>) -> tensor<1x50x24xf32>
    %423 = "mhlo.fusion"(%422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x50x24xf32>, tensor<f32>) -> tensor<1x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %438 = stablehlo.add %436, %437 : tensor<1x50xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x50xf32>
      mhlo.return %439 : tensor<1x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x24xf32>) -> tensor<1x50xf32>
    %424 = "mhlo.fusion"(%arg193, %arg194, %423, %419, %420) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x50xf32>, %arg204: tensor<1x50x768xf32>, %arg205: tensor<1x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %435 = stablehlo.multiply %arg205, %434 : tensor<1x50xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<1x50x768xf32>
      %439 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<50xf32>) -> tensor<1x50x768xf32>
      %441 = stablehlo.multiply %438, %440 : tensor<1x50x768xf32>
      %442 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %443 = stablehlo.multiply %441, %442 : tensor<1x50x768xf32>
      %444 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x50x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x50x768xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      mhlo.return %446 : tensor<50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x50xf32>, tensor<1x50x768xf32>, tensor<1x50xf32>) -> tensor<50x768xf32>
    %425 = stablehlo.dot_general %424, %arg195, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x768xf32>, tensor<3072x768xf32>) -> tensor<50x3072xf32>
    %426 = "mhlo.fusion"(%425, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<3072xf32>):
      %434 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %435 = stablehlo.add %434, %arg201 : tensor<50x3072xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %439 = stablehlo.multiply %436, %438 : tensor<1x50x3072xf32>
      %440 = stablehlo.negate %439 : tensor<1x50x3072xf32>
      %441 = stablehlo.exponential %440 : tensor<1x50x3072xf32>
      %442 = stablehlo.add %437, %441 : tensor<1x50x3072xf32>
      %443 = stablehlo.divide %437, %442 : tensor<1x50x3072xf32>
      %444 = stablehlo.multiply %436, %443 : tensor<1x50x3072xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
      mhlo.return %445 : tensor<50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<3072xf32>) -> tensor<50x3072xf32>
    %427 = stablehlo.dot_general %426, %arg197, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<50x768xf32>
    %428 = "mhlo.fusion"(%419, %427, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x50x768xf32>, %arg202: tensor<50x768xf32>, %arg203: tensor<768xf32>):
      %434 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<50x768xf32>
      %435 = stablehlo.add %434, %arg202 : tensor<50x768xf32>
      %436 = mhlo.bitcast %435 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x768xf32>) -> tensor<1x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<1x50x768xf32>
      %438 = stablehlo.slice %437 [0:1, 1:50, 0:768] : (tensor<1x50x768xf32>) -> tensor<1x49x768xf32>
      mhlo.return %438 : tensor<1x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x50x768xf32>, tensor<50x768xf32>, tensor<768xf32>) -> tensor<1x49x768xf32>
    %429 = "mhlo.fusion"(%428, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x49x768xf32>, %arg202: tensor<f32>):
      %434 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %435 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %435 : tensor<f32>
      }) : (tensor<1x49x768xf32>, tensor<f32>) -> tensor<1x2x768xf32>
      mhlo.return %434 : tensor<1x2x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49x768xf32>, tensor<f32>) -> tensor<1x2x768xf32>
    %430 = "mhlo.fusion"(%429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x2x768xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2x768xf32>, tensor<f32>) -> tensor<1x768xf32>
      %cst_2 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<1x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      mhlo.return %437 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x768xf32>) -> tensor<768xf32>
    %431 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2x768xf32>):
      %434 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,2]{0,1}"} : (tensor<2x768xf32>) -> tensor<768x2xf32>
      %435 = mhlo.copy %434 : tensor<768x2xf32>
      mhlo.return %435 : tensor<768x2xf32>
    }) {output_operand_aliasing = []} : (tensor<2x768xf32>) -> tensor<768x2xf32>
    %432 = "mhlo.fusion"(%arg200, %430, %431) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg201: tensor<2xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768x2xf32>):
      %434 = stablehlo.dot %arg202, %arg203, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x2xf32>) -> tensor<2xf32>
      %435 = stablehlo.add %arg201, %434 : tensor<2xf32>
      mhlo.return %435 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2xf32>, tensor<768xf32>, tensor<768x2xf32>) -> tensor<2xf32>
    %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2xf32>) -> tensor<1x2xf32>
    return %433 : tensor<1x2xf32>
  }
}
