module {
  func.func @main(%arg0: tensor<8x3x224x224xf32>, %arg1: tensor<96x3x4x4xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<96xf32>, %arg5: tensor<96x1x7x7xf32>, %arg6: tensor<96xf32>, %arg7: tensor<96xf32>, %arg8: tensor<96xf32>, %arg9: tensor<384x96xf32>, %arg10: tensor<384xf32>, %arg11: tensor<1x1x1x384xf32>, %arg12: tensor<1x1x1x384xf32>, %arg13: tensor<96x384xf32>, %arg14: tensor<96xf32>, %arg15: tensor<96x1x7x7xf32>, %arg16: tensor<96xf32>, %arg17: tensor<96xf32>, %arg18: tensor<96xf32>, %arg19: tensor<384x96xf32>, %arg20: tensor<384xf32>, %arg21: tensor<1x1x1x384xf32>, %arg22: tensor<1x1x1x384xf32>, %arg23: tensor<96x384xf32>, %arg24: tensor<96xf32>, %arg25: tensor<96x1x7x7xf32>, %arg26: tensor<96xf32>, %arg27: tensor<96xf32>, %arg28: tensor<96xf32>, %arg29: tensor<384x96xf32>, %arg30: tensor<384xf32>, %arg31: tensor<1x1x1x384xf32>, %arg32: tensor<1x1x1x384xf32>, %arg33: tensor<96x384xf32>, %arg34: tensor<96xf32>, %arg35: tensor<96xf32>, %arg36: tensor<96xf32>, %arg37: tensor<192x96x2x2xf32>, %arg38: tensor<192xf32>, %arg39: tensor<192x1x7x7xf32>, %arg40: tensor<192xf32>, %arg41: tensor<192xf32>, %arg42: tensor<192xf32>, %arg43: tensor<768x192xf32>, %arg44: tensor<768xf32>, %arg45: tensor<1x1x1x768xf32>, %arg46: tensor<1x1x1x768xf32>, %arg47: tensor<192x768xf32>, %arg48: tensor<192xf32>, %arg49: tensor<192x1x7x7xf32>, %arg50: tensor<192xf32>, %arg51: tensor<192xf32>, %arg52: tensor<192xf32>, %arg53: tensor<768x192xf32>, %arg54: tensor<768xf32>, %arg55: tensor<1x1x1x768xf32>, %arg56: tensor<1x1x1x768xf32>, %arg57: tensor<192x768xf32>, %arg58: tensor<192xf32>, %arg59: tensor<192x1x7x7xf32>, %arg60: tensor<192xf32>, %arg61: tensor<192xf32>, %arg62: tensor<192xf32>, %arg63: tensor<768x192xf32>, %arg64: tensor<768xf32>, %arg65: tensor<1x1x1x768xf32>, %arg66: tensor<1x1x1x768xf32>, %arg67: tensor<192x768xf32>, %arg68: tensor<192xf32>, %arg69: tensor<192xf32>, %arg70: tensor<192xf32>, %arg71: tensor<384x192x2x2xf32>, %arg72: tensor<384xf32>, %arg73: tensor<384x1x7x7xf32>, %arg74: tensor<384xf32>, %arg75: tensor<384xf32>, %arg76: tensor<384xf32>, %arg77: tensor<1536x384xf32>, %arg78: tensor<1536xf32>, %arg79: tensor<1x1x1x1536xf32>, %arg80: tensor<1x1x1x1536xf32>, %arg81: tensor<384x1536xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384x1x7x7xf32>, %arg84: tensor<384xf32>, %arg85: tensor<384xf32>, %arg86: tensor<384xf32>, %arg87: tensor<1536x384xf32>, %arg88: tensor<1536xf32>, %arg89: tensor<1x1x1x1536xf32>, %arg90: tensor<1x1x1x1536xf32>, %arg91: tensor<384x1536xf32>, %arg92: tensor<384xf32>, %arg93: tensor<384x1x7x7xf32>, %arg94: tensor<384xf32>, %arg95: tensor<384xf32>, %arg96: tensor<384xf32>, %arg97: tensor<1536x384xf32>, %arg98: tensor<1536xf32>, %arg99: tensor<1x1x1x1536xf32>, %arg100: tensor<1x1x1x1536xf32>, %arg101: tensor<384x1536xf32>, %arg102: tensor<384xf32>, %arg103: tensor<384x1x7x7xf32>, %arg104: tensor<384xf32>, %arg105: tensor<384xf32>, %arg106: tensor<384xf32>, %arg107: tensor<1536x384xf32>, %arg108: tensor<1536xf32>, %arg109: tensor<1x1x1x1536xf32>, %arg110: tensor<1x1x1x1536xf32>, %arg111: tensor<384x1536xf32>, %arg112: tensor<384xf32>, %arg113: tensor<384x1x7x7xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<384xf32>, %arg117: tensor<1536x384xf32>, %arg118: tensor<1536xf32>, %arg119: tensor<1x1x1x1536xf32>, %arg120: tensor<1x1x1x1536xf32>, %arg121: tensor<384x1536xf32>, %arg122: tensor<384xf32>, %arg123: tensor<384x1x7x7xf32>, %arg124: tensor<384xf32>, %arg125: tensor<384xf32>, %arg126: tensor<384xf32>, %arg127: tensor<1536x384xf32>, %arg128: tensor<1536xf32>, %arg129: tensor<1x1x1x1536xf32>, %arg130: tensor<1x1x1x1536xf32>, %arg131: tensor<384x1536xf32>, %arg132: tensor<384xf32>, %arg133: tensor<384x1x7x7xf32>, %arg134: tensor<384xf32>, %arg135: tensor<384xf32>, %arg136: tensor<384xf32>, %arg137: tensor<1536x384xf32>, %arg138: tensor<1536xf32>, %arg139: tensor<1x1x1x1536xf32>, %arg140: tensor<1x1x1x1536xf32>, %arg141: tensor<384x1536xf32>, %arg142: tensor<384xf32>, %arg143: tensor<384x1x7x7xf32>, %arg144: tensor<384xf32>, %arg145: tensor<384xf32>, %arg146: tensor<384xf32>, %arg147: tensor<1536x384xf32>, %arg148: tensor<1536xf32>, %arg149: tensor<1x1x1x1536xf32>, %arg150: tensor<1x1x1x1536xf32>, %arg151: tensor<384x1536xf32>, %arg152: tensor<384xf32>, %arg153: tensor<384x1x7x7xf32>, %arg154: tensor<384xf32>, %arg155: tensor<384xf32>, %arg156: tensor<384xf32>, %arg157: tensor<1536x384xf32>, %arg158: tensor<1536xf32>, %arg159: tensor<1x1x1x1536xf32>, %arg160: tensor<1x1x1x1536xf32>, %arg161: tensor<384x1536xf32>, %arg162: tensor<384xf32>, %arg163: tensor<384xf32>, %arg164: tensor<384xf32>, %arg165: tensor<768x384x2x2xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x1x7x7xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<3072x768xf32>, %arg172: tensor<3072xf32>, %arg173: tensor<1x1x1x3072xf32>, %arg174: tensor<1x1x1x3072xf32>, %arg175: tensor<768x3072xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768x1x7x7xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<3072x768xf32>, %arg182: tensor<3072xf32>, %arg183: tensor<1x1x1x3072xf32>, %arg184: tensor<1x1x1x3072xf32>, %arg185: tensor<768x3072xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x1x7x7xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<1x1x1x3072xf32>, %arg194: tensor<1x1x1x3072xf32>, %arg195: tensor<768x3072xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<768xf32>, %arg199: tensor<1000x768xf32>, %arg200: tensor<1000xf32>) -> tensor<8x1000xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x3x224x224xf32>):
      %394 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,224,224,3]{2,1,3,0}"} : (tensor<8x3x224x224xf32>) -> tensor<8x224x224x3xf32>
      %395 = mhlo.copy %394 : tensor<8x224x224x3xf32>
      mhlo.return %395 : tensor<8x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3x224x224xf32>) -> tensor<8x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x3x4x4xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,3,96]{1,0,2,3}"} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
      %395 = mhlo.copy %394 : tensor<4x4x3x96xf32>
      mhlo.return %395 : tensor<4x4x3x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x224x224x3xf32>, tensor<4x4x3x96xf32>) -> tensor<8x56x56x96xf32>
    %3 = "mhlo.fusion"(%2, %arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x56x56x96xf32>
      mhlo.return %395 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %5 = "mhlo.fusion"(%4, %2, %arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>, %arg202: tensor<8x56x56x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x56x56x96xf32>
      mhlo.return %400 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
      mhlo.return %394 : tensor<8x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
    %7 = "mhlo.fusion"(%6, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %8 = "mhlo.fusion"(%7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x56x56xf32>
      mhlo.return %398 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>) -> tensor<8x56x56xf32>
    %9 = "mhlo.fusion"(%arg3, %arg4, %8, %4, %2, %arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<8x56x56xf32>, %arg204: tensor<8x56x56xf32>, %arg205: tensor<8x56x56x96xf32>, %arg206: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x56x56x96xf32>
      mhlo.return %405 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %10 = "mhlo.fusion"(%arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x96xf32>
      mhlo.return %395 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %11 = stablehlo.convolution(%9, %10) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<8x56x56x96xf32>
    %12 = "mhlo.fusion"(%11, %arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x56x56x96xf32>
      mhlo.return %395 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %13 = "mhlo.fusion"(%12, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %14 = "mhlo.fusion"(%13, %11, %arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>, %arg202: tensor<8x56x56x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x56x56x96xf32>
      mhlo.return %400 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %15 = "mhlo.fusion"(%14, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
      mhlo.return %394 : tensor<8x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
    %16 = "mhlo.fusion"(%15, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %17 = "mhlo.fusion"(%16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x56x56xf32>
      mhlo.return %398 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>) -> tensor<8x56x56xf32>
    %18 = "mhlo.fusion"(%arg7, %arg8, %17, %13, %11, %arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<8x56x56xf32>, %arg204: tensor<8x56x56xf32>, %arg205: tensor<8x56x56x96xf32>, %arg206: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x56x56x96xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x96xf32>) -> tensor<25088x96xf32>
      mhlo.return %406 : tensor<25088x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<25088x96xf32>
    %19 = stablehlo.dot_general %18, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x96xf32>, tensor<384x96xf32>) -> tensor<25088x384xf32>
    %20 = "mhlo.fusion"(%19, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<25088x384xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<25088x384xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<25088x384xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %399 = stablehlo.negate %396 : tensor<25088x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<25088x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %403 = stablehlo.abs %402 : tensor<8x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x56x56x384xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x56x56x384xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x56x56x384xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x56x56x384xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x56x56x384xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x56x56x384xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x56x56x384xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x56x56x384xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %431 = stablehlo.negate %407 : tensor<8x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %434 = stablehlo.exponential %431 : tensor<8x56x56x384xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x56x56x384xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x56x56x384xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x56x56x384xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x56x56x384xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x56x56x384xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x56x56x384xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x56x56x384xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x56x56x384xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x56x56x384xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x56x56x384xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x56x56x384xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x56x56x384xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x56x56x384xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x56x56x384xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x56x56x384xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x56x56x384xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x56x56x384xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x56x56x384xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x56x56x384xf32>
      mhlo.return %492 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x384xf32>, tensor<384xf32>) -> tensor<8x56x56x384xf32>
    %21 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x56x56x384xf32>
      mhlo.return %394 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xf32>
    %22 = "mhlo.fusion"(%21, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
      mhlo.return %394 : tensor<8x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
    %23 = "mhlo.fusion"(%22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x2x2x384xf32>, tensor<f32>) -> tensor<8x384xf32>
      %395 = stablehlo.sqrt %394 : tensor<8x384xf32>
      mhlo.return %395 : tensor<8x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x2x384xf32>) -> tensor<8x384xf32>
    %24 = "mhlo.fusion"(%23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      mhlo.return %395 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %25 = "mhlo.fusion"(%24, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
      mhlo.return %394 : tensor<8x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
    %26 = "mhlo.fusion"(%25, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x12xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %27 = "mhlo.fusion"(%26, %23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x384xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x384xf32>
      mhlo.return %402 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %28 = "mhlo.fusion"(%20, %arg11, %27, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<8x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x384xf32>) -> tensor<8x384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x384xf32>) -> tensor<8x56x56x384xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x56x56x384xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x56x56x384xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x56x56x384xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x56x56x384xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x384xf32>) -> tensor<25088x384xf32>
      mhlo.return %404 : tensor<25088x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<8x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<25088x384xf32>
    %29 = stablehlo.dot_general %28, %arg13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x384xf32>, tensor<96x384xf32>) -> tensor<25088x96xf32>
    %30 = "mhlo.fusion"(%29, %arg14, %arg3, %arg4, %8, %4, %2, %arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<96xf32>, %arg205: tensor<8x56x56xf32>, %arg206: tensor<8x56x56xf32>, %arg207: tensor<8x56x56x96xf32>, %arg208: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg207, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg206, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<25088x96xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<25088x96xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x96xf32>) -> tensor<8x56x56x96xf32>
      %409 = stablehlo.add %405, %408 : tensor<8x56x56x96xf32>
      mhlo.return %409 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %31 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x96xf32>
      mhlo.return %395 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %32 = stablehlo.convolution(%30, %31) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<8x56x56x96xf32>
    %33 = "mhlo.fusion"(%32, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x56x56x96xf32>
      mhlo.return %395 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %34 = "mhlo.fusion"(%33, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %35 = "mhlo.fusion"(%34, %32, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>, %arg202: tensor<8x56x56x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x56x56x96xf32>
      mhlo.return %400 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %36 = "mhlo.fusion"(%35, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
      mhlo.return %394 : tensor<8x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
    %37 = "mhlo.fusion"(%36, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %38 = "mhlo.fusion"(%37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x56x56xf32>
      mhlo.return %398 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>) -> tensor<8x56x56xf32>
    %39 = "mhlo.fusion"(%arg17, %arg18, %38, %34, %32, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<8x56x56xf32>, %arg204: tensor<8x56x56xf32>, %arg205: tensor<8x56x56x96xf32>, %arg206: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x56x56x96xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x96xf32>) -> tensor<25088x96xf32>
      mhlo.return %406 : tensor<25088x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<25088x96xf32>
    %40 = stablehlo.dot_general %39, %arg19, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x96xf32>, tensor<384x96xf32>) -> tensor<25088x384xf32>
    %41 = "mhlo.fusion"(%40, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<25088x384xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<25088x384xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<25088x384xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %399 = stablehlo.negate %396 : tensor<25088x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<25088x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %403 = stablehlo.abs %402 : tensor<8x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x56x56x384xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x56x56x384xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x56x56x384xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x56x56x384xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x56x56x384xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x56x56x384xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x56x56x384xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x56x56x384xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %431 = stablehlo.negate %407 : tensor<8x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %434 = stablehlo.exponential %431 : tensor<8x56x56x384xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x56x56x384xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x56x56x384xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x56x56x384xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x56x56x384xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x56x56x384xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x56x56x384xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x56x56x384xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x56x56x384xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x56x56x384xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x56x56x384xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x56x56x384xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x56x56x384xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x56x56x384xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x56x56x384xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x56x56x384xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x56x56x384xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x56x56x384xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x56x56x384xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x56x56x384xf32>
      mhlo.return %492 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x384xf32>, tensor<384xf32>) -> tensor<8x56x56x384xf32>
    %42 = "mhlo.fusion"(%41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x56x56x384xf32>
      mhlo.return %394 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xf32>
    %43 = "mhlo.fusion"(%42, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
      mhlo.return %394 : tensor<8x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
    %44 = "mhlo.fusion"(%43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x2x2x384xf32>, tensor<f32>) -> tensor<8x384xf32>
      %395 = stablehlo.sqrt %394 : tensor<8x384xf32>
      mhlo.return %395 : tensor<8x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x2x384xf32>) -> tensor<8x384xf32>
    %45 = "mhlo.fusion"(%44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      mhlo.return %395 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %46 = "mhlo.fusion"(%45, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
      mhlo.return %394 : tensor<8x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
    %47 = "mhlo.fusion"(%46, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x12xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %48 = "mhlo.fusion"(%47, %44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x384xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x384xf32>
      mhlo.return %402 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %49 = "mhlo.fusion"(%41, %arg21, %48, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<8x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x384xf32>) -> tensor<8x384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x384xf32>) -> tensor<8x56x56x384xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x56x56x384xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x56x56x384xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x56x56x384xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x56x56x384xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x384xf32>) -> tensor<25088x384xf32>
      mhlo.return %404 : tensor<25088x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<8x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<25088x384xf32>
    %50 = stablehlo.dot_general %49, %arg23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x384xf32>, tensor<96x384xf32>) -> tensor<25088x96xf32>
    %51 = "mhlo.fusion"(%30, %50, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<25088x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<25088x96xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<25088x96xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x96xf32>) -> tensor<8x56x56x96xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x56x56x96xf32>
      mhlo.return %397 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<25088x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %52 = "mhlo.fusion"(%arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x96xf32>
      mhlo.return %395 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %53 = stablehlo.convolution(%51, %52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<8x56x56x96xf32>
    %54 = "mhlo.fusion"(%53, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x56x56x96xf32>
      mhlo.return %395 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %55 = "mhlo.fusion"(%54, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %56 = "mhlo.fusion"(%55, %53, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>, %arg202: tensor<8x56x56x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x56x56x96xf32>
      mhlo.return %400 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %57 = "mhlo.fusion"(%56, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
      mhlo.return %394 : tensor<8x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
    %58 = "mhlo.fusion"(%57, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x56x56xf32>
      mhlo.return %398 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>) -> tensor<8x56x56xf32>
    %60 = "mhlo.fusion"(%arg27, %arg28, %59, %55, %53, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<8x56x56xf32>, %arg204: tensor<8x56x56xf32>, %arg205: tensor<8x56x56x96xf32>, %arg206: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x56x56x96xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x96xf32>) -> tensor<25088x96xf32>
      mhlo.return %406 : tensor<25088x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<96xf32>) -> tensor<25088x96xf32>
    %61 = stablehlo.dot_general %60, %arg29, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x96xf32>, tensor<384x96xf32>) -> tensor<25088x384xf32>
    %62 = "mhlo.fusion"(%61, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<25088x384xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<25088x384xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<25088x384xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %399 = stablehlo.negate %396 : tensor<25088x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<25088x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x384xf32>) -> tensor<8x56x56x384xf32>
      %403 = stablehlo.abs %402 : tensor<8x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x56x56x384xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x56x56x384xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x56x56x384xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x56x56x384xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x56x56x384xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x56x56x384xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x56x56x384xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x56x56x384xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %431 = stablehlo.negate %407 : tensor<8x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %434 = stablehlo.exponential %431 : tensor<8x56x56x384xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x56x56x384xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x56x56x384xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x56x56x384xf32>, tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x56x56x384xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x56x56x384xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x56x56x384xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x56x56x384xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x56x56x384xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x56x56x384xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x56x56x384xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x56x56x384xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x56x56x384xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x56x56x384xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x56x56x384xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x56x56x384xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x56x56x384xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x56x56x384xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x56x56x384xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x56x56x384xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x56x56x384xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x56x56x384xi1>, tensor<8x56x56x384xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x56x56x384xf32>
      mhlo.return %492 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x384xf32>, tensor<384xf32>) -> tensor<8x56x56x384xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x56x56x384xf32>
      mhlo.return %394 : tensor<8x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>) -> tensor<8x56x56x384xf32>
    %64 = "mhlo.fusion"(%63, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
      mhlo.return %394 : tensor<8x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<f32>) -> tensor<8x2x2x384xf32>
    %65 = "mhlo.fusion"(%64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x2x2x384xf32>, tensor<f32>) -> tensor<8x384xf32>
      %395 = stablehlo.sqrt %394 : tensor<8x384xf32>
      mhlo.return %395 : tensor<8x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x2x384xf32>) -> tensor<8x384xf32>
    %66 = "mhlo.fusion"(%65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      mhlo.return %395 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %67 = "mhlo.fusion"(%66, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
      mhlo.return %394 : tensor<8x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x384xf32>, tensor<f32>) -> tensor<8x1x1x12xf32>
    %68 = "mhlo.fusion"(%67, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x12xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x12xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %69 = "mhlo.fusion"(%68, %65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x384xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,384]{3,0,2,1}"} : (tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x384xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x384xf32>
      mhlo.return %402 : tensor<8x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x384xf32>) -> tensor<8x1x1x384xf32>
    %70 = "mhlo.fusion"(%62, %arg31, %69, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<8x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x384xf32>) -> tensor<8x384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x384xf32>) -> tensor<8x56x56x384xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x56x56x384xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x56x56x384xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<384xf32>) -> tensor<8x56x56x384xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x56x56x384xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x56x56x384xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x56x56x384xf32>) -> tensor<25088x384xf32>
      mhlo.return %404 : tensor<25088x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<8x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<25088x384xf32>
    %71 = stablehlo.dot_general %70, %arg33, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x384xf32>, tensor<96x384xf32>) -> tensor<25088x96xf32>
    %72 = "mhlo.fusion"(%51, %71, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<25088x96xf32>, %arg203: tensor<96xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<25088x96xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<25088x96xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x96xf32>) -> tensor<8x56x56x96xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x56x56x96xf32>
      mhlo.return %397 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<25088x96xf32>, tensor<96xf32>) -> tensor<8x56x56x96xf32>
    %73 = "mhlo.fusion"(%72, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %74 = "mhlo.fusion"(%72, %73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg202, %394 : tensor<8x56x56xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %397 = stablehlo.subtract %arg201, %396 : tensor<8x56x56x96xf32>
      %398 = stablehlo.multiply %397, %397 : tensor<8x56x56x96xf32>
      mhlo.return %398 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
    %75 = "mhlo.fusion"(%74, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
      mhlo.return %394 : tensor<8x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x96xf32>, tensor<f32>) -> tensor<8x56x56x3xf32>
    %76 = "mhlo.fusion"(%75, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x56x56x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
      mhlo.return %394 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56x3xf32>, tensor<f32>) -> tensor<8x56x56xf32>
    %77 = "mhlo.fusion"(%76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x56x56xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x56x56xf32>
      mhlo.return %398 : tensor<8x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<8x56x56xf32>) -> tensor<8x56x56xf32>
    %78 = "mhlo.fusion"(%arg35, %arg36, %77, %72, %73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<8x56x56xf32>, %arg204: tensor<8x56x56x96xf32>, %arg205: tensor<8x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x56x56xf32>
      %395 = stablehlo.multiply %arg205, %394 : tensor<8x56x56xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %397 = stablehlo.subtract %arg204, %396 : tensor<8x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<8x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<8x56x56x96xf32>
      %403 = stablehlo.add %401, %402 : tensor<8x56x56x96xf32>
      mhlo.return %403 : tensor<8x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<8x56x56xf32>, tensor<8x56x56x96xf32>, tensor<8x56x56xf32>) -> tensor<8x56x56x96xf32>
    %79 = "mhlo.fusion"(%arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x96x2x2xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,96,192]{1,0,2,3}"} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
      %395 = mhlo.copy %394 : tensor<2x2x96x192xf32>
      mhlo.return %395 : tensor<2x2x96x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
    %80 = stablehlo.convolution(%78, %79) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x56x56x96xf32>, tensor<2x2x96x192xf32>) -> tensor<8x28x28x192xf32>
    %81 = "mhlo.fusion"(%80, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x28x28x192xf32>
      mhlo.return %395 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %82 = "mhlo.fusion"(%arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x192xf32>
      mhlo.return %395 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %83 = stablehlo.convolution(%81, %82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<8x28x28x192xf32>
    %84 = "mhlo.fusion"(%83, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x28x28x192xf32>
      mhlo.return %395 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %85 = "mhlo.fusion"(%84, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %86 = "mhlo.fusion"(%85, %83, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>, %arg202: tensor<8x28x28x192xf32>, %arg203: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x28x28x192xf32>
      mhlo.return %400 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %87 = "mhlo.fusion"(%86, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
      mhlo.return %394 : tensor<8x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
    %88 = "mhlo.fusion"(%87, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x6xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x28x28xf32>
      mhlo.return %398 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>) -> tensor<8x28x28xf32>
    %90 = "mhlo.fusion"(%arg41, %arg42, %89, %85, %83, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<8x28x28xf32>, %arg204: tensor<8x28x28xf32>, %arg205: tensor<8x28x28x192xf32>, %arg206: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x28x28x192xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x28x28x192xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x192xf32>) -> tensor<6272x192xf32>
      mhlo.return %406 : tensor<6272x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x28x28xf32>, tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<6272x192xf32>
    %91 = stablehlo.dot_general %90, %arg43, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x192xf32>, tensor<768x192xf32>) -> tensor<6272x768xf32>
    %92 = "mhlo.fusion"(%91, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6272x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6272x768xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<6272x768xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<6272x768xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %399 = stablehlo.negate %396 : tensor<6272x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<6272x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %403 = stablehlo.abs %402 : tensor<8x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x28x28x768xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x28x28x768xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x28x28x768xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x28x28x768xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x28x28x768xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x28x28x768xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x28x28x768xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x28x28x768xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %431 = stablehlo.negate %407 : tensor<8x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %434 = stablehlo.exponential %431 : tensor<8x28x28x768xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x28x28x768xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x28x28x768xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x28x28x768xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x28x28x768xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x28x28x768xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x28x28x768xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x28x28x768xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x28x28x768xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x28x28x768xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x28x28x768xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x28x28x768xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x28x28x768xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x28x28x768xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x28x28x768xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x28x28x768xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x28x28x768xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x28x28x768xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x28x28x768xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x28x28x768xf32>
      mhlo.return %492 : tensor<8x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<6272x768xf32>, tensor<768xf32>) -> tensor<8x28x28x768xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x28x28x768xf32>, tensor<f32>) -> tensor<8x768xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x768xf32>
      mhlo.return %396 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>) -> tensor<8x768xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      mhlo.return %395 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %95 = "mhlo.fusion"(%94, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
      mhlo.return %394 : tensor<8x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
    %96 = "mhlo.fusion"(%95, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x24xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %97 = "mhlo.fusion"(%96, %93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x768xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x768xf32>
      mhlo.return %402 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %98 = "mhlo.fusion"(%92, %arg45, %97, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<8x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x768xf32>) -> tensor<8x768xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x768xf32>) -> tensor<8x28x28x768xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x28x28x768xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x28x28x768xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x28x28x768xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x28x28x768xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x768xf32>) -> tensor<6272x768xf32>
      mhlo.return %404 : tensor<6272x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<8x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<6272x768xf32>
    %99 = stablehlo.dot_general %98, %arg47, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x768xf32>, tensor<192x768xf32>) -> tensor<6272x192xf32>
    %100 = "mhlo.fusion"(%99, %arg48, %80, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6272x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<8x28x28x192xf32>, %arg204: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg203, %394 : tensor<8x28x28x192xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<6272x192xf32>
      %397 = stablehlo.add %396, %arg201 : tensor<6272x192xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x192xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.add %395, %398 : tensor<8x28x28x192xf32>
      mhlo.return %399 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6272x192xf32>, tensor<192xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %101 = "mhlo.fusion"(%arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x192xf32>
      mhlo.return %395 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %102 = stablehlo.convolution(%100, %101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<8x28x28x192xf32>
    %103 = "mhlo.fusion"(%102, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x28x28x192xf32>
      mhlo.return %395 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %104 = "mhlo.fusion"(%103, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %105 = "mhlo.fusion"(%104, %102, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>, %arg202: tensor<8x28x28x192xf32>, %arg203: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x28x28x192xf32>
      mhlo.return %400 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
      mhlo.return %394 : tensor<8x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x6xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %108 = "mhlo.fusion"(%107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x28x28xf32>
      mhlo.return %398 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>) -> tensor<8x28x28xf32>
    %109 = "mhlo.fusion"(%arg51, %arg52, %108, %104, %102, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<8x28x28xf32>, %arg204: tensor<8x28x28xf32>, %arg205: tensor<8x28x28x192xf32>, %arg206: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x28x28x192xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x28x28x192xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x192xf32>) -> tensor<6272x192xf32>
      mhlo.return %406 : tensor<6272x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x28x28xf32>, tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<6272x192xf32>
    %110 = stablehlo.dot_general %109, %arg53, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x192xf32>, tensor<768x192xf32>) -> tensor<6272x768xf32>
    %111 = "mhlo.fusion"(%110, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6272x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6272x768xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<6272x768xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<6272x768xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %399 = stablehlo.negate %396 : tensor<6272x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<6272x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %403 = stablehlo.abs %402 : tensor<8x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x28x28x768xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x28x28x768xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x28x28x768xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x28x28x768xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x28x28x768xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x28x28x768xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x28x28x768xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x28x28x768xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %431 = stablehlo.negate %407 : tensor<8x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %434 = stablehlo.exponential %431 : tensor<8x28x28x768xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x28x28x768xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x28x28x768xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x28x28x768xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x28x28x768xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x28x28x768xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x28x28x768xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x28x28x768xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x28x28x768xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x28x28x768xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x28x28x768xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x28x28x768xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x28x28x768xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x28x28x768xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x28x28x768xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x28x28x768xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x28x28x768xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x28x28x768xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x28x28x768xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x28x28x768xf32>
      mhlo.return %492 : tensor<8x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<6272x768xf32>, tensor<768xf32>) -> tensor<8x28x28x768xf32>
    %112 = "mhlo.fusion"(%111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x28x28x768xf32>, tensor<f32>) -> tensor<8x768xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x768xf32>
      mhlo.return %396 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>) -> tensor<8x768xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      mhlo.return %395 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %114 = "mhlo.fusion"(%113, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
      mhlo.return %394 : tensor<8x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
    %115 = "mhlo.fusion"(%114, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x24xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %116 = "mhlo.fusion"(%115, %112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x768xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x768xf32>
      mhlo.return %402 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %117 = "mhlo.fusion"(%111, %arg55, %116, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<8x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x768xf32>) -> tensor<8x768xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x768xf32>) -> tensor<8x28x28x768xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x28x28x768xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x28x28x768xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x28x28x768xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x28x28x768xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x768xf32>) -> tensor<6272x768xf32>
      mhlo.return %404 : tensor<6272x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<8x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<6272x768xf32>
    %118 = stablehlo.dot_general %117, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x768xf32>, tensor<192x768xf32>) -> tensor<6272x192xf32>
    %119 = "mhlo.fusion"(%100, %118, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<6272x192xf32>, %arg203: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<6272x192xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<6272x192xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x192xf32>) -> tensor<8x28x28x192xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x28x28x192xf32>
      mhlo.return %397 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<6272x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %120 = "mhlo.fusion"(%arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x192xf32>
      mhlo.return %395 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %121 = stablehlo.convolution(%119, %120) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<8x28x28x192xf32>
    %122 = "mhlo.fusion"(%121, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x28x28x192xf32>
      mhlo.return %395 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %123 = "mhlo.fusion"(%122, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %124 = "mhlo.fusion"(%123, %121, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>, %arg202: tensor<8x28x28x192xf32>, %arg203: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x28x28x192xf32>
      mhlo.return %400 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
      mhlo.return %394 : tensor<8x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x6xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x28x28xf32>
      mhlo.return %398 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>) -> tensor<8x28x28xf32>
    %128 = "mhlo.fusion"(%arg61, %arg62, %127, %123, %121, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<8x28x28xf32>, %arg204: tensor<8x28x28xf32>, %arg205: tensor<8x28x28x192xf32>, %arg206: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x28x28x192xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x28x28x192xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x192xf32>) -> tensor<6272x192xf32>
      mhlo.return %406 : tensor<6272x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x28x28xf32>, tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<192xf32>) -> tensor<6272x192xf32>
    %129 = stablehlo.dot_general %128, %arg63, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x192xf32>, tensor<768x192xf32>) -> tensor<6272x768xf32>
    %130 = "mhlo.fusion"(%129, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6272x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6272x768xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<6272x768xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<6272x768xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %399 = stablehlo.negate %396 : tensor<6272x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6272x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<6272x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x768xf32>) -> tensor<8x28x28x768xf32>
      %403 = stablehlo.abs %402 : tensor<8x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x28x28x768xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x28x28x768xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x28x28x768xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x28x28x768xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x28x28x768xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x28x28x768xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x28x28x768xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x28x28x768xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %431 = stablehlo.negate %407 : tensor<8x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %434 = stablehlo.exponential %431 : tensor<8x28x28x768xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x28x28x768xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x28x28x768xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x28x28x768xf32>, tensor<8x28x28x768xf32>) -> tensor<8x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x28x28x768xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x28x28x768xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x28x28x768xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x28x28x768xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x28x28x768xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x28x28x768xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x28x28x768xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x28x28x768xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x28x28x768xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x28x28x768xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x28x28x768xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x28x28x768xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x28x28x768xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x28x28x768xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x28x28x768xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x28x28x768xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x28x28x768xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x28x28x768xi1>, tensor<8x28x28x768xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x28x28x768xf32>
      mhlo.return %492 : tensor<8x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<6272x768xf32>, tensor<768xf32>) -> tensor<8x28x28x768xf32>
    %131 = "mhlo.fusion"(%130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x28x28x768xf32>, tensor<f32>) -> tensor<8x768xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x768xf32>
      mhlo.return %396 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>) -> tensor<8x768xf32>
    %132 = "mhlo.fusion"(%131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      mhlo.return %395 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
      mhlo.return %394 : tensor<8x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x768xf32>, tensor<f32>) -> tensor<8x1x1x24xf32>
    %134 = "mhlo.fusion"(%133, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x24xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x24xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %135 = "mhlo.fusion"(%134, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x768xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,768]{3,0,2,1}"} : (tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x768xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x768xf32>
      mhlo.return %402 : tensor<8x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x768xf32>) -> tensor<8x1x1x768xf32>
    %136 = "mhlo.fusion"(%130, %arg65, %135, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<8x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x768xf32>) -> tensor<8x768xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x768xf32>) -> tensor<8x28x28x768xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x28x28x768xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x28x28x768xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<768xf32>) -> tensor<8x28x28x768xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x28x28x768xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x28x28x768xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x28x28x768xf32>) -> tensor<6272x768xf32>
      mhlo.return %404 : tensor<6272x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<8x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<6272x768xf32>
    %137 = stablehlo.dot_general %136, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6272x768xf32>, tensor<192x768xf32>) -> tensor<6272x192xf32>
    %138 = "mhlo.fusion"(%119, %137, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<6272x192xf32>, %arg203: tensor<192xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<6272x192xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<6272x192xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6272x192xf32>) -> tensor<8x28x28x192xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x28x28x192xf32>
      mhlo.return %397 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<6272x192xf32>, tensor<192xf32>) -> tensor<8x28x28x192xf32>
    %139 = "mhlo.fusion"(%138, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %140 = "mhlo.fusion"(%138, %139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg202, %394 : tensor<8x28x28xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %397 = stablehlo.subtract %arg201, %396 : tensor<8x28x28x192xf32>
      %398 = stablehlo.multiply %397, %397 : tensor<8x28x28x192xf32>
      mhlo.return %398 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
    %141 = "mhlo.fusion"(%140, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28x192xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
      mhlo.return %394 : tensor<8x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x192xf32>, tensor<f32>) -> tensor<8x28x28x6xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x28x28x6xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
      mhlo.return %394 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28x6xf32>, tensor<f32>) -> tensor<8x28x28xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %397 = stablehlo.add %395, %396 : tensor<8x28x28xf32>
      %398 = stablehlo.rsqrt %397 : tensor<8x28x28xf32>
      mhlo.return %398 : tensor<8x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<8x28x28xf32>) -> tensor<8x28x28xf32>
    %144 = "mhlo.fusion"(%arg69, %arg70, %143, %138, %139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<8x28x28xf32>, %arg204: tensor<8x28x28x192xf32>, %arg205: tensor<8x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x28x28xf32>
      %395 = stablehlo.multiply %arg205, %394 : tensor<8x28x28xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %397 = stablehlo.subtract %arg204, %396 : tensor<8x28x28x192xf32>
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<8x28x28x192xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x28x28x192xf32>
      %402 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<8x28x28x192xf32>
      %403 = stablehlo.add %401, %402 : tensor<8x28x28x192xf32>
      mhlo.return %403 : tensor<8x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x28x28xf32>, tensor<8x28x28x192xf32>, tensor<8x28x28xf32>) -> tensor<8x28x28x192xf32>
    %145 = "mhlo.fusion"(%arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x192x2x2xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,192,384]{1,0,2,3}"} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
      %395 = mhlo.copy %394 : tensor<2x2x192x384xf32>
      mhlo.return %395 : tensor<2x2x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
    %146 = stablehlo.convolution(%144, %145) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x28x28x192xf32>, tensor<2x2x192x384xf32>) -> tensor<8x14x14x384xf32>
    %147 = "mhlo.fusion"(%146, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %148 = "mhlo.fusion"(%arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %149 = stablehlo.convolution(%147, %148) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %150 = "mhlo.fusion"(%149, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %151 = "mhlo.fusion"(%150, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %152 = "mhlo.fusion"(%151, %149, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %155 = "mhlo.fusion"(%arg75, %arg76, %154, %151, %149, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %156 = stablehlo.dot_general %155, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %157 = "mhlo.fusion"(%156, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %158 = "mhlo.fusion"(%157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %160 = "mhlo.fusion"(%159, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %161 = "mhlo.fusion"(%160, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %162 = "mhlo.fusion"(%161, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %163 = "mhlo.fusion"(%162, %158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %164 = "mhlo.fusion"(%157, %arg79, %163, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %165 = stablehlo.dot_general %164, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %166 = "mhlo.fusion"(%165, %arg82, %146, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14x384xf32>, %arg204: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg203, %394 : tensor<8x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %397 = stablehlo.add %396, %arg201 : tensor<1568x384xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.add %395, %398 : tensor<8x14x14x384xf32>
      mhlo.return %399 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x384xf32>, tensor<384xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %167 = "mhlo.fusion"(%arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %168 = stablehlo.convolution(%166, %167) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %169 = "mhlo.fusion"(%168, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %170 = "mhlo.fusion"(%169, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %171 = "mhlo.fusion"(%170, %168, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %172 = "mhlo.fusion"(%171, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %174 = "mhlo.fusion"(%arg85, %arg86, %173, %170, %168, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %175 = stablehlo.dot_general %174, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %176 = "mhlo.fusion"(%175, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %178 = "mhlo.fusion"(%177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %179 = "mhlo.fusion"(%178, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %180 = "mhlo.fusion"(%179, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %181 = "mhlo.fusion"(%180, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %182 = "mhlo.fusion"(%181, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %183 = "mhlo.fusion"(%176, %arg89, %182, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %184 = stablehlo.dot_general %183, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %185 = "mhlo.fusion"(%166, %184, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %186 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %187 = stablehlo.convolution(%185, %186) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %188 = "mhlo.fusion"(%187, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %189 = "mhlo.fusion"(%188, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %190 = "mhlo.fusion"(%189, %187, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %191 = "mhlo.fusion"(%190, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %192 = "mhlo.fusion"(%191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %193 = "mhlo.fusion"(%arg95, %arg96, %192, %189, %187, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %194 = stablehlo.dot_general %193, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %195 = "mhlo.fusion"(%194, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %198 = "mhlo.fusion"(%197, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %200 = "mhlo.fusion"(%199, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %201 = "mhlo.fusion"(%200, %196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %202 = "mhlo.fusion"(%195, %arg99, %201, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %203 = stablehlo.dot_general %202, %arg101, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %204 = "mhlo.fusion"(%185, %203, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %205 = "mhlo.fusion"(%arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %206 = stablehlo.convolution(%204, %205) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %207 = "mhlo.fusion"(%206, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %208 = "mhlo.fusion"(%207, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %209 = "mhlo.fusion"(%208, %206, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %212 = "mhlo.fusion"(%arg105, %arg106, %211, %208, %206, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %213 = stablehlo.dot_general %212, %arg107, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %214 = "mhlo.fusion"(%213, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %216 = "mhlo.fusion"(%215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %217 = "mhlo.fusion"(%216, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %218 = "mhlo.fusion"(%217, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %219 = "mhlo.fusion"(%218, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %220 = "mhlo.fusion"(%219, %215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %221 = "mhlo.fusion"(%214, %arg109, %220, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %222 = stablehlo.dot_general %221, %arg111, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %223 = "mhlo.fusion"(%204, %222, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %224 = "mhlo.fusion"(%arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %225 = stablehlo.convolution(%223, %224) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %226 = "mhlo.fusion"(%225, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %227 = "mhlo.fusion"(%226, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %228 = "mhlo.fusion"(%227, %225, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %229 = "mhlo.fusion"(%228, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %231 = "mhlo.fusion"(%arg115, %arg116, %230, %227, %225, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %232 = stablehlo.dot_general %231, %arg117, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %233 = "mhlo.fusion"(%232, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %234 = "mhlo.fusion"(%233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %236 = "mhlo.fusion"(%235, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %237 = "mhlo.fusion"(%236, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %238 = "mhlo.fusion"(%237, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %239 = "mhlo.fusion"(%238, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %240 = "mhlo.fusion"(%233, %arg119, %239, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %241 = stablehlo.dot_general %240, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %242 = "mhlo.fusion"(%223, %241, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %243 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %244 = stablehlo.convolution(%242, %243) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %245 = "mhlo.fusion"(%244, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %246 = "mhlo.fusion"(%245, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %247 = "mhlo.fusion"(%246, %244, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %248 = "mhlo.fusion"(%247, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %249 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %250 = "mhlo.fusion"(%arg125, %arg126, %249, %246, %244, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %251 = stablehlo.dot_general %250, %arg127, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %252 = "mhlo.fusion"(%251, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %253 = "mhlo.fusion"(%252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %254 = "mhlo.fusion"(%253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %255 = "mhlo.fusion"(%254, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %257 = "mhlo.fusion"(%256, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %258 = "mhlo.fusion"(%257, %253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %259 = "mhlo.fusion"(%252, %arg129, %258, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %260 = stablehlo.dot_general %259, %arg131, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %261 = "mhlo.fusion"(%242, %260, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %262 = "mhlo.fusion"(%arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %263 = stablehlo.convolution(%261, %262) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %264 = "mhlo.fusion"(%263, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %265 = "mhlo.fusion"(%264, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %266 = "mhlo.fusion"(%265, %263, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %269 = "mhlo.fusion"(%arg135, %arg136, %268, %265, %263, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %270 = stablehlo.dot_general %269, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %271 = "mhlo.fusion"(%270, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %272 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %273 = "mhlo.fusion"(%272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %275 = "mhlo.fusion"(%274, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %277 = "mhlo.fusion"(%276, %272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %278 = "mhlo.fusion"(%271, %arg139, %277, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %279 = stablehlo.dot_general %278, %arg141, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %280 = "mhlo.fusion"(%261, %279, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %281 = "mhlo.fusion"(%arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %282 = stablehlo.convolution(%280, %281) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %283 = "mhlo.fusion"(%282, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %284 = "mhlo.fusion"(%283, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %285 = "mhlo.fusion"(%284, %282, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %286 = "mhlo.fusion"(%285, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %288 = "mhlo.fusion"(%arg145, %arg146, %287, %284, %282, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %289 = stablehlo.dot_general %288, %arg147, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %290 = "mhlo.fusion"(%289, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %292 = "mhlo.fusion"(%291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %293 = "mhlo.fusion"(%292, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %294 = "mhlo.fusion"(%293, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %295 = "mhlo.fusion"(%294, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %296 = "mhlo.fusion"(%295, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %297 = "mhlo.fusion"(%290, %arg149, %296, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %298 = stablehlo.dot_general %297, %arg151, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %299 = "mhlo.fusion"(%280, %298, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %300 = "mhlo.fusion"(%arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x384xf32>
      mhlo.return %395 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %301 = stablehlo.convolution(%299, %300) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<8x14x14x384xf32>
    %302 = "mhlo.fusion"(%301, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x14x14x384xf32>
      mhlo.return %395 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %304 = "mhlo.fusion"(%303, %301, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14xf32>, %arg202: tensor<8x14x14x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x14x14x384xf32>
      mhlo.return %400 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %305 = "mhlo.fusion"(%304, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %306 = "mhlo.fusion"(%305) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %307 = "mhlo.fusion"(%arg155, %arg156, %306, %303, %301, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14xf32>, %arg205: tensor<8x14x14x384xf32>, %arg206: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x14x14xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x14x14x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x384xf32>) -> tensor<1568x384xf32>
      mhlo.return %406 : tensor<1568x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<384xf32>) -> tensor<1568x384xf32>
    %308 = stablehlo.dot_general %307, %arg157, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x384xf32>, tensor<1536x384xf32>) -> tensor<1568x1536xf32>
    %309 = "mhlo.fusion"(%308, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1568x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<1568x1536xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<1568x1536xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<1568x1536xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %399 = stablehlo.negate %396 : tensor<1568x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1568x1536xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<1568x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x1536xf32>) -> tensor<8x14x14x1536xf32>
      %403 = stablehlo.abs %402 : tensor<8x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x14x14x1536xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x14x14x1536xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x14x14x1536xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x14x14x1536xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x14x14x1536xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x14x14x1536xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x14x14x1536xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x14x14x1536xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %431 = stablehlo.negate %407 : tensor<8x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %434 = stablehlo.exponential %431 : tensor<8x14x14x1536xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x14x14x1536xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x14x14x1536xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x14x14x1536xf32>, tensor<8x14x14x1536xf32>) -> tensor<8x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x14x14x1536xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x14x14x1536xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x14x14x1536xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x14x14x1536xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x14x14x1536xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x14x14x1536xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x14x14x1536xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x14x14x1536xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x14x14x1536xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x14x14x1536xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x14x14x1536xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x14x14x1536xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x14x14x1536xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x14x14x1536xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x14x14x1536xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x14x14x1536xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x14x14x1536xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x14x14x1536xi1>, tensor<8x14x14x1536xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x14x14x1536xf32>
      mhlo.return %492 : tensor<8x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1568x1536xf32>, tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
    %310 = "mhlo.fusion"(%309) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x14x14x1536xf32>, tensor<f32>) -> tensor<8x1536xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x1536xf32>
      mhlo.return %396 : tensor<8x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>) -> tensor<8x1536xf32>
    %311 = "mhlo.fusion"(%310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      mhlo.return %395 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x1536xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
      mhlo.return %394 : tensor<8x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x1536xf32>, tensor<f32>) -> tensor<8x1x1x48xf32>
    %313 = "mhlo.fusion"(%312, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x48xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
      mhlo.return %394 : tensor<8x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x48xf32>, tensor<f32>) -> tensor<8x1x1x2xf32>
    %314 = "mhlo.fusion"(%313, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x2xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x2xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %315 = "mhlo.fusion"(%314, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x1536xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,1536]{3,0,2,1}"} : (tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1536xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x1536xf32>
      mhlo.return %402 : tensor<8x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x1536xf32>) -> tensor<8x1x1x1536xf32>
    %316 = "mhlo.fusion"(%309, %arg159, %315, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<8x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x1536xf32>) -> tensor<8x1536xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x1536xf32>) -> tensor<8x14x14x1536xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x14x14x1536xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x14x14x1536xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<1536xf32>) -> tensor<8x14x14x1536xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x14x14x1536xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x14x14x1536xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x14x14x1536xf32>) -> tensor<1568x1536xf32>
      mhlo.return %404 : tensor<1568x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<8x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<1568x1536xf32>
    %317 = stablehlo.dot_general %316, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1568x1536xf32>, tensor<384x1536xf32>) -> tensor<1568x384xf32>
    %318 = "mhlo.fusion"(%299, %317, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<1568x384xf32>, %arg203: tensor<384xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1568x384xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<1568x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1568x384xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x14x14x384xf32>
      mhlo.return %397 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<1568x384xf32>, tensor<384xf32>) -> tensor<8x14x14x384xf32>
    %319 = "mhlo.fusion"(%318, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      mhlo.return %394 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14xf32>
    %320 = "mhlo.fusion"(%318, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<8x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %395 = stablehlo.multiply %arg202, %394 : tensor<8x14x14xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.subtract %arg201, %396 : tensor<8x14x14x384xf32>
      %398 = stablehlo.multiply %397, %397 : tensor<8x14x14x384xf32>
      mhlo.return %398 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
    %321 = "mhlo.fusion"(%320, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x384xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
      mhlo.return %394 : tensor<8x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x384xf32>, tensor<f32>) -> tensor<8x14x14x12xf32>
    %322 = "mhlo.fusion"(%321) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x14x14x12xf32>, tensor<f32>) -> tensor<8x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x14x14xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x14x14xf32>
      mhlo.return %399 : tensor<8x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<8x14x14x12xf32>) -> tensor<8x14x14xf32>
    %323 = "mhlo.fusion"(%arg163, %arg164, %322, %318, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<8x14x14xf32>, %arg204: tensor<8x14x14x384xf32>, %arg205: tensor<8x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x14x14xf32>
      %395 = stablehlo.multiply %arg205, %394 : tensor<8x14x14xf32>
      %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %397 = stablehlo.subtract %arg204, %396 : tensor<8x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<8x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<8x14x14x384xf32>
      %403 = stablehlo.add %401, %402 : tensor<8x14x14x384xf32>
      mhlo.return %403 : tensor<8x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x14x14xf32>, tensor<8x14x14x384xf32>, tensor<8x14x14xf32>) -> tensor<8x14x14x384xf32>
    %324 = "mhlo.fusion"(%arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x384x2x2xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,384,768]{1,0,2,3}"} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
      %395 = mhlo.copy %394 : tensor<2x2x384x768xf32>
      mhlo.return %395 : tensor<2x2x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
    %325 = stablehlo.convolution(%323, %324) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x14x14x384xf32>, tensor<2x2x384x768xf32>) -> tensor<8x7x7x768xf32>
    %326 = "mhlo.fusion"(%325, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x7x7x768xf32>
      mhlo.return %395 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %327 = "mhlo.fusion"(%arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x768xf32>
      mhlo.return %395 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %328 = stablehlo.convolution(%326, %327) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<8x7x7x768xf32>
    %329 = "mhlo.fusion"(%328, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x7x7x768xf32>
      mhlo.return %395 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %330 = "mhlo.fusion"(%329, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      mhlo.return %394 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
    %331 = "mhlo.fusion"(%330, %328, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7xf32>, %arg202: tensor<8x7x7x768xf32>, %arg203: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x7x7x768xf32>
      mhlo.return %400 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %332 = "mhlo.fusion"(%331, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
      mhlo.return %394 : tensor<8x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
    %333 = "mhlo.fusion"(%332) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x24xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x7x7xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x7x7xf32>
      mhlo.return %399 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x24xf32>) -> tensor<8x7x7xf32>
    %334 = "mhlo.fusion"(%arg169, %arg170, %333, %330, %328, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x7x7xf32>, %arg204: tensor<8x7x7xf32>, %arg205: tensor<8x7x7x768xf32>, %arg206: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x7x7x768xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x7x7x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x768xf32>) -> tensor<392x768xf32>
      mhlo.return %406 : tensor<392x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x7x7xf32>, tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<392x768xf32>
    %335 = stablehlo.dot_general %334, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x768xf32>, tensor<3072x768xf32>) -> tensor<392x3072xf32>
    %336 = "mhlo.fusion"(%335, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<392x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<392x3072xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<392x3072xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<392x3072xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %399 = stablehlo.negate %396 : tensor<392x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<392x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %403 = stablehlo.abs %402 : tensor<8x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x7x7x3072xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x7x7x3072xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x7x7x3072xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x7x7x3072xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x7x7x3072xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x7x7x3072xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x7x7x3072xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x7x7x3072xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %431 = stablehlo.negate %407 : tensor<8x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %434 = stablehlo.exponential %431 : tensor<8x7x7x3072xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x7x7x3072xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x7x7x3072xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x7x7x3072xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x7x7x3072xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x7x7x3072xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x7x7x3072xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x7x7x3072xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x7x7x3072xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x7x7x3072xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x7x7x3072xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x7x7x3072xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x7x7x3072xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x7x7x3072xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x7x7x3072xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x7x7x3072xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x7x7x3072xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x7x7x3072xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x7x7x3072xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x7x7x3072xf32>
      mhlo.return %492 : tensor<8x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<392x3072xf32>, tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
    %337 = "mhlo.fusion"(%336) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x7x7x3072xf32>, tensor<f32>) -> tensor<8x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x3072xf32>
      mhlo.return %396 : tensor<8x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>) -> tensor<8x3072xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      mhlo.return %395 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %339 = "mhlo.fusion"(%338, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3072xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
      mhlo.return %394 : tensor<8x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
    %340 = "mhlo.fusion"(%339, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
      mhlo.return %394 : tensor<8x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
    %341 = "mhlo.fusion"(%340, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %342 = "mhlo.fusion"(%341, %337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x3072xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x3072xf32>
      mhlo.return %402 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %343 = "mhlo.fusion"(%336, %arg173, %342, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<8x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x3072xf32>) -> tensor<8x3072xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x3072xf32>) -> tensor<8x7x7x3072xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x7x7x3072xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x7x7x3072xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x7x7x3072xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x7x7x3072xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x3072xf32>) -> tensor<392x3072xf32>
      mhlo.return %404 : tensor<392x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<8x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<392x3072xf32>
    %344 = stablehlo.dot_general %343, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x3072xf32>, tensor<768x3072xf32>) -> tensor<392x768xf32>
    %345 = "mhlo.fusion"(%344, %arg176, %325, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<392x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x7x7x768xf32>, %arg204: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg203, %394 : tensor<8x7x7x768xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<392x768xf32>
      %397 = stablehlo.add %396, %arg201 : tensor<392x768xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x768xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.add %395, %398 : tensor<8x7x7x768xf32>
      mhlo.return %399 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<392x768xf32>, tensor<768xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %346 = "mhlo.fusion"(%arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x768xf32>
      mhlo.return %395 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %347 = stablehlo.convolution(%345, %346) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<8x7x7x768xf32>
    %348 = "mhlo.fusion"(%347, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x7x7x768xf32>
      mhlo.return %395 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %349 = "mhlo.fusion"(%348, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      mhlo.return %394 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
    %350 = "mhlo.fusion"(%349, %347, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7xf32>, %arg202: tensor<8x7x7x768xf32>, %arg203: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x7x7x768xf32>
      mhlo.return %400 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %351 = "mhlo.fusion"(%350, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
      mhlo.return %394 : tensor<8x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
    %352 = "mhlo.fusion"(%351) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x24xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x7x7xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x7x7xf32>
      mhlo.return %399 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x24xf32>) -> tensor<8x7x7xf32>
    %353 = "mhlo.fusion"(%arg179, %arg180, %352, %349, %347, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x7x7xf32>, %arg204: tensor<8x7x7xf32>, %arg205: tensor<8x7x7x768xf32>, %arg206: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x7x7x768xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x7x7x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x768xf32>) -> tensor<392x768xf32>
      mhlo.return %406 : tensor<392x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x7x7xf32>, tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<392x768xf32>
    %354 = stablehlo.dot_general %353, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x768xf32>, tensor<3072x768xf32>) -> tensor<392x3072xf32>
    %355 = "mhlo.fusion"(%354, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<392x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<392x3072xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<392x3072xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<392x3072xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %399 = stablehlo.negate %396 : tensor<392x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<392x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %403 = stablehlo.abs %402 : tensor<8x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x7x7x3072xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x7x7x3072xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x7x7x3072xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x7x7x3072xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x7x7x3072xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x7x7x3072xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x7x7x3072xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x7x7x3072xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %431 = stablehlo.negate %407 : tensor<8x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %434 = stablehlo.exponential %431 : tensor<8x7x7x3072xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x7x7x3072xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x7x7x3072xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x7x7x3072xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x7x7x3072xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x7x7x3072xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x7x7x3072xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x7x7x3072xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x7x7x3072xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x7x7x3072xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x7x7x3072xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x7x7x3072xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x7x7x3072xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x7x7x3072xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x7x7x3072xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x7x7x3072xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x7x7x3072xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x7x7x3072xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x7x7x3072xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x7x7x3072xf32>
      mhlo.return %492 : tensor<8x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<392x3072xf32>, tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
    %356 = "mhlo.fusion"(%355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x7x7x3072xf32>, tensor<f32>) -> tensor<8x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x3072xf32>
      mhlo.return %396 : tensor<8x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>) -> tensor<8x3072xf32>
    %357 = "mhlo.fusion"(%356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      mhlo.return %395 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %358 = "mhlo.fusion"(%357, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3072xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
      mhlo.return %394 : tensor<8x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
    %359 = "mhlo.fusion"(%358, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
      mhlo.return %394 : tensor<8x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
    %360 = "mhlo.fusion"(%359, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %361 = "mhlo.fusion"(%360, %356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x3072xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x3072xf32>
      mhlo.return %402 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %362 = "mhlo.fusion"(%355, %arg183, %361, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<8x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x3072xf32>) -> tensor<8x3072xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x3072xf32>) -> tensor<8x7x7x3072xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x7x7x3072xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x7x7x3072xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x7x7x3072xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x7x7x3072xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x3072xf32>) -> tensor<392x3072xf32>
      mhlo.return %404 : tensor<392x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<8x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<392x3072xf32>
    %363 = stablehlo.dot_general %362, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x3072xf32>, tensor<768x3072xf32>) -> tensor<392x768xf32>
    %364 = "mhlo.fusion"(%345, %363, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<392x768xf32>, %arg203: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<392x768xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<392x768xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x768xf32>) -> tensor<8x7x7x768xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x7x7x768xf32>
      mhlo.return %397 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<392x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %365 = "mhlo.fusion"(%arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %394 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %395 = mhlo.copy %394 : tensor<7x7x1x768xf32>
      mhlo.return %395 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %366 = stablehlo.convolution(%364, %365) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<8x7x7x768xf32>
    %367 = "mhlo.fusion"(%366, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg201, %394 : tensor<8x7x7x768xf32>
      mhlo.return %395 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %368 = "mhlo.fusion"(%367, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      mhlo.return %394 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7xf32>
    %369 = "mhlo.fusion"(%368, %366, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7xf32>, %arg202: tensor<8x7x7x768xf32>, %arg203: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg202, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x7x7x768xf32>
      mhlo.return %400 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %370 = "mhlo.fusion"(%369, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
      mhlo.return %394 : tensor<8x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x7x7x24xf32>
    %371 = "mhlo.fusion"(%370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<8x7x7x24xf32>, tensor<f32>) -> tensor<8x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %398 = stablehlo.add %396, %397 : tensor<8x7x7xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8x7x7xf32>
      mhlo.return %399 : tensor<8x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x24xf32>) -> tensor<8x7x7xf32>
    %372 = "mhlo.fusion"(%arg189, %arg190, %371, %368, %366, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8x7x7xf32>, %arg204: tensor<8x7x7xf32>, %arg205: tensor<8x7x7x768xf32>, %arg206: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %395 = stablehlo.add %arg205, %394 : tensor<8x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x7x7xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8x7x7xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x7x7x768xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<8x7x7xf32>) -> tensor<8x7x7x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x7x7x768xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<8x7x7x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x7x7x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x768xf32>) -> tensor<392x768xf32>
      mhlo.return %406 : tensor<392x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x7x7xf32>, tensor<8x7x7xf32>, tensor<8x7x7x768xf32>, tensor<768xf32>) -> tensor<392x768xf32>
    %373 = stablehlo.dot_general %372, %arg191, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x768xf32>, tensor<3072x768xf32>) -> tensor<392x3072xf32>
    %374 = "mhlo.fusion"(%373, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<392x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %395 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<392x3072xf32>
      %396 = stablehlo.add %395, %arg201 : tensor<392x3072xf32>
      %397 = stablehlo.multiply %394, %396 : tensor<392x3072xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %399 = stablehlo.negate %396 : tensor<392x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<392x3072xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<392x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x3072xf32>) -> tensor<8x7x7x3072xf32>
      %403 = stablehlo.abs %402 : tensor<8x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %405 = stablehlo.compare LT, %403, %404 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %407 = stablehlo.multiply %402, %402 : tensor<8x7x7x3072xf32>
      %408 = stablehlo.multiply %406, %407 : tensor<8x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x7x7x3072xf32>
      %411 = stablehlo.multiply %410, %407 : tensor<8x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %413 = stablehlo.add %411, %412 : tensor<8x7x7x3072xf32>
      %414 = stablehlo.multiply %413, %407 : tensor<8x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x7x7x3072xf32>
      %417 = stablehlo.multiply %416, %407 : tensor<8x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %419 = stablehlo.add %417, %418 : tensor<8x7x7x3072xf32>
      %420 = stablehlo.multiply %419, %407 : tensor<8x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x7x7x3072xf32>
      %423 = stablehlo.multiply %422, %407 : tensor<8x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %425 = stablehlo.add %423, %424 : tensor<8x7x7x3072xf32>
      %426 = stablehlo.multiply %402, %425 : tensor<8x7x7x3072xf32>
      %427 = stablehlo.subtract %404, %426 : tensor<8x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %429 = stablehlo.compare LT, %402, %428 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %431 = stablehlo.negate %407 : tensor<8x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %434 = stablehlo.exponential %431 : tensor<8x7x7x3072xf32>
      %435 = stablehlo.divide %404, %403 : tensor<8x7x7x3072xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<8x7x7x3072xf32>
      %437 = stablehlo.compare LT, %403, %430 : (tensor<8x7x7x3072xf32>, tensor<8x7x7x3072xf32>) -> tensor<8x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %439 = stablehlo.divide %404, %407 : tensor<8x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<8x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<8x7x7x3072xf32>
      %443 = stablehlo.multiply %442, %439 : tensor<8x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<8x7x7x3072xf32>
      %446 = stablehlo.multiply %445, %439 : tensor<8x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<8x7x7x3072xf32>
      %449 = stablehlo.multiply %448, %439 : tensor<8x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<8x7x7x3072xf32>
      %452 = stablehlo.multiply %451, %439 : tensor<8x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %454 = stablehlo.add %452, %453 : tensor<8x7x7x3072xf32>
      %455 = stablehlo.multiply %454, %439 : tensor<8x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %457 = stablehlo.add %455, %456 : tensor<8x7x7x3072xf32>
      %458 = stablehlo.multiply %457, %439 : tensor<8x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %460 = stablehlo.add %458, %459 : tensor<8x7x7x3072xf32>
      %461 = stablehlo.multiply %460, %439 : tensor<8x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %439 : tensor<8x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<8x7x7x3072xf32>
      %468 = stablehlo.multiply %467, %439 : tensor<8x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %470 = stablehlo.add %468, %469 : tensor<8x7x7x3072xf32>
      %471 = stablehlo.multiply %470, %439 : tensor<8x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %473 = stablehlo.add %471, %472 : tensor<8x7x7x3072xf32>
      %474 = stablehlo.multiply %473, %439 : tensor<8x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %476 = stablehlo.add %474, %475 : tensor<8x7x7x3072xf32>
      %477 = stablehlo.multiply %476, %439 : tensor<8x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %479 = stablehlo.add %477, %478 : tensor<8x7x7x3072xf32>
      %480 = stablehlo.multiply %479, %439 : tensor<8x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %482 = stablehlo.add %480, %481 : tensor<8x7x7x3072xf32>
      %483 = stablehlo.multiply %482, %439 : tensor<8x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x7x7x3072xf32>
      %485 = stablehlo.add %483, %484 : tensor<8x7x7x3072xf32>
      %486 = stablehlo.select %437, %463, %485 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %487 = stablehlo.multiply %436, %486 : tensor<8x7x7x3072xf32>
      %488 = stablehlo.select %433, %428, %487 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %489 = stablehlo.subtract %430, %488 : tensor<8x7x7x3072xf32>
      %490 = stablehlo.select %429, %489, %488 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %491 = stablehlo.select %405, %427, %490 : tensor<8x7x7x3072xi1>, tensor<8x7x7x3072xf32>
      %492 = stablehlo.multiply %398, %491 : tensor<8x7x7x3072xf32>
      mhlo.return %492 : tensor<8x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<392x3072xf32>, tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
    %375 = "mhlo.fusion"(%374) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>):
      %394 = stablehlo.multiply %arg201, %arg201 : tensor<8x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %395 = stablehlo.reduce(%394 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x7x7x3072xf32>, tensor<f32>) -> tensor<8x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<8x3072xf32>
      mhlo.return %396 : tensor<8x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>) -> tensor<8x3072xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      mhlo.return %395 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %377 = "mhlo.fusion"(%376, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3072xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
      mhlo.return %394 : tensor<8x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3072xf32>, tensor<f32>) -> tensor<8x1x1x96xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x96xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
      mhlo.return %394 : tensor<8x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x96xf32>, tensor<f32>) -> tensor<8x1x1x3xf32>
    %379 = "mhlo.fusion"(%378, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1x3xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
      mhlo.return %394 : tensor<8x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1x3xf32>, tensor<f32>) -> tensor<8x1x1xf32>
    %380 = "mhlo.fusion"(%379, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1x1xf32>, %arg202: tensor<8x3072xf32>):
      %394 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,1,3072]{3,0,2,1}"} : (tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
      %395 = mhlo.copy %394 : tensor<8x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1xf32>
      %399 = stablehlo.add %397, %398 : tensor<8x1x1xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x1xf32>) -> tensor<8xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x3072xf32>
      %402 = stablehlo.divide %395, %401 : tensor<8x1x1x3072xf32>
      mhlo.return %402 : tensor<8x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x1xf32>, tensor<8x3072xf32>) -> tensor<8x1x1x3072xf32>
    %381 = "mhlo.fusion"(%374, %arg193, %380, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<8x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %394 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %396 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x1x3072xf32>) -> tensor<8x3072xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [0, 3] : (tensor<8x3072xf32>) -> tensor<8x7x7x3072xf32>
      %398 = stablehlo.multiply %arg201, %397 : tensor<8x7x7x3072xf32>
      %399 = stablehlo.multiply %395, %398 : tensor<8x7x7x3072xf32>
      %400 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [3] : (tensor<3072xf32>) -> tensor<8x7x7x3072xf32>
      %402 = stablehlo.add %399, %401 : tensor<8x7x7x3072xf32>
      %403 = stablehlo.add %402, %arg201 : tensor<8x7x7x3072xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x7x7x3072xf32>) -> tensor<392x3072xf32>
      mhlo.return %404 : tensor<392x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<8x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<392x3072xf32>
    %382 = stablehlo.dot_general %381, %arg195, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<392x3072xf32>, tensor<768x3072xf32>) -> tensor<392x768xf32>
    %383 = "mhlo.fusion"(%364, %382, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<392x768xf32>, %arg203: tensor<768xf32>):
      %394 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<392x768xf32>
      %395 = stablehlo.add %394, %arg202 : tensor<392x768xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<392x768xf32>) -> tensor<8x7x7x768xf32>
      %397 = stablehlo.add %arg201, %396 : tensor<8x7x7x768xf32>
      mhlo.return %397 : tensor<8x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<392x768xf32>, tensor<768xf32>) -> tensor<8x7x7x768xf32>
    %384 = "mhlo.fusion"(%383, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<8x7x7x768xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x768xf32>
      mhlo.return %394 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x7x7x768xf32>, tensor<f32>) -> tensor<8x768xf32>
    %385 = "mhlo.fusion"(%384) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768xf32>
      %395 = stablehlo.multiply %arg201, %394 : tensor<8x768xf32>
      mhlo.return %395 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>) -> tensor<8x768xf32>
    %386 = "mhlo.fusion"(%385, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x768xf32>, tensor<f32>) -> tensor<8x24xf32>
      mhlo.return %394 : tensor<8x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>, tensor<f32>) -> tensor<8x24xf32>
    %387 = "mhlo.fusion"(%386, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x24xf32>, %arg202: tensor<f32>):
      %394 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1] : (tensor<8x24xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %394 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x24xf32>, tensor<f32>) -> tensor<8xf32>
    %388 = "mhlo.fusion"(%387, %384) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8xf32>, %arg202: tensor<8x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768xf32>
      %395 = stablehlo.multiply %arg202, %394 : tensor<8x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %397 = stablehlo.multiply %arg201, %396 : tensor<8xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0] : (tensor<8xf32>) -> tensor<8x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x768xf32>
      %400 = stablehlo.multiply %399, %399 : tensor<8x768xf32>
      mhlo.return %400 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<8xf32>, tensor<8x768xf32>) -> tensor<8x768xf32>
    %389 = "mhlo.fusion"(%388, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x768xf32>, %arg202: tensor<f32>):
      %394 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %395 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %395 : tensor<f32>
      }) : (tensor<8x768xf32>, tensor<f32>) -> tensor<8x24xf32>
      mhlo.return %394 : tensor<8x24xf32>
    }) {output_operand_aliasing = []} : (tensor<8x768xf32>, tensor<f32>) -> tensor<8x24xf32>
    %390 = "mhlo.fusion"(%389) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %394 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x24xf32>, tensor<f32>) -> tensor<8xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<8xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %398 = stablehlo.add %396, %397 : tensor<8xf32>
      %399 = stablehlo.rsqrt %398 : tensor<8xf32>
      mhlo.return %399 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x24xf32>) -> tensor<8xf32>
    %391 = "mhlo.fusion"(%arg197, %arg198, %390, %387, %384) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<8xf32>, %arg204: tensor<8xf32>, %arg205: tensor<8x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768xf32>
      %395 = stablehlo.multiply %arg205, %394 : tensor<8x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %397 = stablehlo.multiply %arg204, %396 : tensor<8xf32>
      %398 = stablehlo.broadcast_in_dim %397, dims = [0] : (tensor<8xf32>) -> tensor<8x768xf32>
      %399 = stablehlo.subtract %395, %398 : tensor<8x768xf32>
      %400 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<8xf32>) -> tensor<8x768xf32>
      %401 = stablehlo.multiply %399, %400 : tensor<8x768xf32>
      %402 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<8x768xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<8x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<768xf32>) -> tensor<8x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<8x768xf32>
      mhlo.return %405 : tensor<8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x768xf32>) -> tensor<8x768xf32>
    %392 = stablehlo.dot_general %391, %arg199, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x768xf32>, tensor<1000x768xf32>) -> tensor<8x1000xf32>
    %393 = "mhlo.fusion"(%392, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<8x1000xf32>, %arg202: tensor<1000xf32>):
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1000xf32>) -> tensor<8x1000xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<8x1000xf32>
      mhlo.return %395 : tensor<8x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1000xf32>, tensor<1000xf32>) -> tensor<8x1000xf32>
    return %393 : tensor<8x1000xf32>
  }
}
