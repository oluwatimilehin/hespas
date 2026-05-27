module {
  func.func @main(%arg0: tensor<256xf32>, %arg1: tensor<8x3x512x512xf32>, %arg2: tensor<32x3x7x7xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<32xf32>, %arg7: tensor<32xf32>, %arg8: tensor<32x32xf32>, %arg9: tensor<32xf32>, %arg10: tensor<32x32x8x8xf32>, %arg11: tensor<32xf32>, %arg12: tensor<32xf32>, %arg13: tensor<32xf32>, %arg14: tensor<32x32xf32>, %arg15: tensor<32xf32>, %arg16: tensor<32x32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<128x32xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128x1x3x3xf32>, %arg25: tensor<128xf32>, %arg26: tensor<32x128xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32x32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<32x32x8x8xf32>, %arg33: tensor<32xf32>, %arg34: tensor<32xf32>, %arg35: tensor<32xf32>, %arg36: tensor<32x32xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32x32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<128x32xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x1x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<32x128xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<256x32xf32>, %arg53: tensor<64x32x3x3xf32>, %arg54: tensor<64xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<64xf32>, %arg58: tensor<64xf32>, %arg59: tensor<64x64xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64x64x4x4xf32>, %arg62: tensor<64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<64x64xf32>, %arg66: tensor<64xf32>, %arg67: tensor<64x64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<64x64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64xf32>, %arg73: tensor<256x64xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256x1x3x3xf32>, %arg76: tensor<256xf32>, %arg77: tensor<64x256xf32>, %arg78: tensor<64xf32>, %arg79: tensor<64xf32>, %arg80: tensor<64xf32>, %arg81: tensor<64x64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64x64x4x4xf32>, %arg84: tensor<64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64xf32>, %arg87: tensor<64x64xf32>, %arg88: tensor<64xf32>, %arg89: tensor<64x64xf32>, %arg90: tensor<64xf32>, %arg91: tensor<64x64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64xf32>, %arg95: tensor<256x64xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256x1x3x3xf32>, %arg98: tensor<256xf32>, %arg99: tensor<64x256xf32>, %arg100: tensor<64xf32>, %arg101: tensor<64xf32>, %arg102: tensor<64xf32>, %arg103: tensor<256x64xf32>, %arg104: tensor<160x64x3x3xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160x160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<160x160xf32>, %arg121: tensor<160xf32>, %arg122: tensor<160xf32>, %arg123: tensor<160xf32>, %arg124: tensor<640x160xf32>, %arg125: tensor<640xf32>, %arg126: tensor<640x1x3x3xf32>, %arg127: tensor<640xf32>, %arg128: tensor<160x640xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160x160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<160x160xf32>, %arg143: tensor<160xf32>, %arg144: tensor<160xf32>, %arg145: tensor<160xf32>, %arg146: tensor<640x160xf32>, %arg147: tensor<640xf32>, %arg148: tensor<640x1x3x3xf32>, %arg149: tensor<640xf32>, %arg150: tensor<160x640xf32>, %arg151: tensor<160xf32>, %arg152: tensor<160xf32>, %arg153: tensor<160xf32>, %arg154: tensor<256x160xf32>, %arg155: tensor<256x160x3x3xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<256xf32>, %arg161: tensor<256x256xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256x256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<256x256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256x256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<1024x256xf32>, %arg172: tensor<1024xf32>, %arg173: tensor<1024x1x3x3xf32>, %arg174: tensor<1024xf32>, %arg175: tensor<256x1024xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256x256xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256x256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<256x256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256x256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<1024x256xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<1024x1x3x3xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<256x1024xf32>, %arg194: tensor<256xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256x256xf32>, %arg198: tensor<256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256x1024x1x1xf32>, %arg203: tensor<256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<150x256x1x1xf32>, %arg207: tensor<150xf32>) -> tensor<8x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<256xf32>
      %409 = stablehlo.rsqrt %408 : tensor<256xf32>
      mhlo.return %409 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<127> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %409 = stablehlo.iota dim = 0 : tensor<128xi32>
      %410 = stablehlo.convert %409 : (tensor<128xi32>) -> tensor<128xf32>
      %411 = stablehlo.maximum %408, %410 : tensor<128xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %413 = stablehlo.convert %412 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_4 = stablehlo.constant dense<1> : tensor<i32>
      %414 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %415 = stablehlo.add %413, %414 : tensor<128x1xi32>
      %416 = stablehlo.minimum %407, %415 : tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %417 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %418 = stablehlo.compare LT, %416, %417 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<128> : tensor<i32>
      %419 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %420 = stablehlo.add %416, %419 : tensor<128x1xi32>
      %421 = stablehlo.select %418, %420, %416 : tensor<128x1xi1>, tensor<128x1xi32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %424 = stablehlo.convert %411 : (tensor<128xf32>) -> tensor<128xi32>
      %425 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %427 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %428 = stablehlo.add %424, %427 : tensor<128xi32>
      %429 = stablehlo.select %426, %428, %424 : tensor<128xi1>, tensor<128xi32>
      %430 = stablehlo.broadcast_in_dim %429, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %431 = stablehlo.concatenate %423, %430, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %431 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %408 = stablehlo.iota dim = 0 : tensor<128xi32>
      %409 = stablehlo.convert %408 : (tensor<128xi32>) -> tensor<128xf32>
      %410 = stablehlo.maximum %407, %409 : tensor<128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %412 = stablehlo.convert %411 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_3 = stablehlo.constant dense<0> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %414 = stablehlo.compare LT, %412, %413 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_4 = stablehlo.constant dense<128> : tensor<i32>
      %415 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %416 = stablehlo.add %412, %415 : tensor<128x1xi32>
      %417 = stablehlo.select %414, %416, %412 : tensor<128x1xi1>, tensor<128x1xi32>
      %418 = mhlo.bitcast %417 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %420 = stablehlo.convert %410 : (tensor<128xf32>) -> tensor<128xi32>
      %421 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %422 = stablehlo.compare LT, %420, %421 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %423 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %424 = stablehlo.add %420, %423 : tensor<128xi32>
      %425 = stablehlo.select %422, %424, %420 : tensor<128xi1>, tensor<128xi32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %427 = stablehlo.concatenate %419, %426, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %427 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x3x512x512xf32>):
      %407 = stablehlo.transpose %arg208, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,512,512,3]{2,1,3,0}"} : (tensor<8x3x512x512xf32>) -> tensor<8x512x512x3xf32>
      %408 = mhlo.copy %407 : tensor<8x512x512x3xf32>
      mhlo.return %408 : tensor<8x512x512x3xf32>
    }) {output_operand_aliasing = []} : (tensor<8x3x512x512xf32>) -> tensor<8x512x512x3xf32>
    %4 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x3x7x7xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,32]{1,0,2,3}"} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
      %408 = mhlo.copy %407 : tensor<7x7x3x32xf32>
      mhlo.return %408 : tensor<7x7x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
    %5 = stablehlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x512x512x3xf32>, tensor<7x7x3x32xf32>) -> tensor<8x128x128x32xf32>
    %6 = "mhlo.fusion"(%5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x128x128x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      mhlo.return %413 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = "mhlo.fusion"(%6, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %8 = "mhlo.fusion"(%7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384xf32>, %arg209: tensor<8x128x128x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x16384xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x16384x32xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %419 = stablehlo.reduce(%418 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %420 = stablehlo.multiply %419, %414 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x16384xf32>
      %423 = stablehlo.rsqrt %422 : tensor<8x16384xf32>
      mhlo.return %423 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<8x16384xf32>
    %9 = "mhlo.fusion"(%arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384xf32>, %arg212: tensor<8x128x128x32xf32>, %arg213: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x16384xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x16384x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x16384x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x16384x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x16384x32xf32>
      mhlo.return %423 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %10 = "mhlo.fusion"(%9, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %11 = "mhlo.fusion"(%10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<8x16384xf32>, %arg212: tensor<8x16384xf32>, %arg213: tensor<8x128x128x32xf32>, %arg214: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg213, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.multiply %arg212, %414 : tensor<8x16384xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x16384x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x16384x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x16384x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x16384x32xf32>
      %424 = stablehlo.multiply %arg208, %414 : tensor<8x16384xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x16384x32xf32>
      %427 = stablehlo.multiply %426, %426 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %428 = stablehlo.reduce(%427 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %429 = stablehlo.multiply %428, %414 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x16384xf32>
      %432 = stablehlo.rsqrt %431 : tensor<8x16384xf32>
      mhlo.return %432 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<8x16384xf32>
    %12 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<8x16384xf32>, %arg215: tensor<8x16384xf32>, %arg216: tensor<8x128x128x32xf32>, %arg217: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x16384xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x16384x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x16384x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x16384x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x16384x32xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x16384xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x16384x32xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x16384x32xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x16384x32xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x16384x32xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
      mhlo.return %433 : tensor<131072x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<131072x32xf32>
    %13 = stablehlo.dot_general %12, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<32x32xf32>) -> tensor<131072x32xf32>
    %14 = "mhlo.fusion"(%13, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      mhlo.return %409 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %15 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<8x16384xf32>, %arg215: tensor<8x16384xf32>, %arg216: tensor<8x128x128x32xf32>, %arg217: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x128x128x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x32x128x128xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %413 = mhlo.copy %412 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x16384xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x16384x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x16384x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x16384x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x16384x32xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x16384xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x16384x32xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x16384x32xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x16384x32xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x16384x32xf32>
      %433 = stablehlo.transpose %432, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,32,16384]{1,2,0}"} : (tensor<8x16384x32xf32>) -> tensor<8x32x16384xf32>
      %434 = mhlo.copy %433 : tensor<8x32x16384xf32>
      %435 = mhlo.bitcast %434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x32x16384xf32>) -> tensor<8x32x128x128xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,32]{2,1,3,0}"} : (tensor<8x32x128x128xf32>) -> tensor<8x128x128x32xf32>
      %437 = mhlo.copy %436 : tensor<8x128x128x32xf32>
      mhlo.return %437 : tensor<8x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>) -> tensor<8x128x128x32xf32>
    %16 = "mhlo.fusion"(%arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %408 = mhlo.copy %407 : tensor<8x8x32x32xf32>
      mhlo.return %408 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %17 = stablehlo.convolution(%15, %16) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<8x16x16x32xf32>
    %18 = "mhlo.fusion"(%17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      mhlo.return %413 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<8x256x32xf32>
    %19 = "mhlo.fusion"(%18, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
    %20 = "mhlo.fusion"(%19, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x32xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %419 = stablehlo.reduce(%418 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
      %420 = stablehlo.multiply %419, %414 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x256xf32>
      %423 = stablehlo.rsqrt %422 : tensor<8x256xf32>
      mhlo.return %423 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<8x256xf32>
    %21 = "mhlo.fusion"(%arg12, %arg13, %20, %19, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x32xf32>, %arg213: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x256x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x256x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x32xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x32xf32>) -> tensor<2048x32xf32>
      mhlo.return %424 : tensor<2048x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<2048x32xf32>
    %22 = stablehlo.dot_general %21, %arg14, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x32xf32>, tensor<32x32xf32>) -> tensor<2048x32xf32>
    %23 = "mhlo.fusion"(%22, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2048x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,256,32]{3,2,0,1}"} : (tensor<2048x32xf32>) -> tensor<8x1x256x32xf32>
      %410 = mhlo.copy %409 : tensor<8x1x256x32xf32>
      %411 = stablehlo.transpose %410, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[8,1,32,256]{2,3,1,0}"} : (tensor<8x1x256x32xf32>) -> tensor<8x1x32x256xf32>
      %412 = mhlo.copy %411 : tensor<8x1x32x256xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x32x256xf32>) -> tensor<8x32x256xf32>
      mhlo.return %413 : tensor<8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x32xf32>, tensor<32xf32>) -> tensor<8x32x256xf32>
    %24 = stablehlo.dot_general %14, %23, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16384x32xf32>, tensor<8x32x256xf32>) -> tensor<8x16384x256xf32>
    %25 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x16384x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[8,1,16384,256]{3,2,0,1}"} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
      %410 = mhlo.copy %409 : tensor<8x1x16384x256xf32>
      mhlo.return %410 : tensor<8x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
    %26 = "mhlo.fusion"(%cst, %25) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<8x1x16384x256xf32>):
      %407 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %409 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %409 : tensor<f32>
      }) : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384x8xf32>
      %408 = stablehlo.reduce(%407 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<8x1x16384x8xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
      mhlo.return %408 : tensor<8x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<8x1x16384x256xf32>) -> tensor<8x1x16384xf32>
    %27 = "mhlo.fusion"(%26, %24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1x16384xf32>, %arg209: tensor<8x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[8,1,16384,256]{3,2,0,1}"} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
      %410 = mhlo.copy %409 : tensor<8x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1x16384xf32>
      %412 = stablehlo.maximum %411, %arg208 : tensor<8x1x16384xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x16384xf32>) -> tensor<8x16384xf32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [0, 2] : (tensor<8x16384xf32>) -> tensor<8x1x16384x256xf32>
      %415 = stablehlo.subtract %410, %414 : tensor<8x1x16384x256xf32>
      %416 = stablehlo.exponential %415 : tensor<8x1x16384x256xf32>
      mhlo.return %416 : tensor<8x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384xf32>, tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
    %28 = "mhlo.fusion"(%27, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1x16384x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
      mhlo.return %407 : tensor<8x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
    %29 = "mhlo.fusion"(%27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1x16384x256xf32>, %arg209: tensor<8x1x16384xf32>):
      %407 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x16384xf32>) -> tensor<8x16384xf32>
      %408 = stablehlo.broadcast_in_dim %407, dims = [0, 2] : (tensor<8x16384xf32>) -> tensor<8x1x16384x256xf32>
      %409 = stablehlo.divide %arg208, %408 : tensor<8x1x16384x256xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x16384x256xf32>) -> tensor<8x16384x256xf32>
      mhlo.return %410 : tensor<8x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384x256xf32>, tensor<8x1x16384xf32>) -> tensor<8x16384x256xf32>
    %30 = stablehlo.dot_general %21, %arg16, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x32xf32>, tensor<32x32xf32>) -> tensor<2048x32xf32>
    %31 = "mhlo.fusion"(%30, %arg17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2048x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %409 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x32xf32>, tensor<32xf32>) -> tensor<8x256x32xf32>
    %32 = stablehlo.dot_general %29, %31, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16384x256xf32>, tensor<8x256x32xf32>) -> tensor<8x16384x32xf32>
    %33 = mhlo.bitcast %32 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
    %34 = stablehlo.dot_general %33, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<32x32xf32>) -> tensor<131072x32xf32>
    %35 = "mhlo.fusion"(%34, %arg4, %arg5, %8, %7, %5, %arg3, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<8x16384xf32>, %arg212: tensor<8x16384xf32>, %arg213: tensor<8x128x128x32xf32>, %arg214: tensor<32xf32>, %arg215: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<8x128x128x32xf32>
      %411 = stablehlo.add %arg213, %410 : tensor<8x128x128x32xf32>
      %412 = stablehlo.transpose %411, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %413 = mhlo.copy %412 : tensor<8x32x128x128xf32>
      %414 = mhlo.bitcast %413 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      %415 = stablehlo.transpose %414, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,32]{1,2,0}"} : (tensor<8x32x16384xf32>) -> tensor<8x16384x32xf32>
      %416 = mhlo.copy %415 : tensor<8x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<8x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %420 = stablehlo.subtract %416, %419 : tensor<8x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<8x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<8x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<8x16384x32xf32>
      %427 = stablehlo.add %409, %426 : tensor<8x16384x32xf32>
      mhlo.return %427 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384xf32>, tensor<8x128x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %36 = "mhlo.fusion"(%35, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %37 = "mhlo.fusion"(%35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.reduce(%411 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %413 = stablehlo.multiply %412, %407 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.add %413, %414 : tensor<8x16384xf32>
      %416 = stablehlo.rsqrt %415 : tensor<8x16384xf32>
      mhlo.return %416 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x16384xf32>
    %38 = "mhlo.fusion"(%arg20, %arg21, %37, %35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
      mhlo.return %417 : tensor<131072x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<131072x32xf32>
    %39 = stablehlo.dot_general %38, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<128x32xf32>) -> tensor<131072x128xf32>
    %40 = "mhlo.fusion"(%39, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x128xf32>, %arg209: tensor<128xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<131072x128xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<131072x128xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x128xf32>) -> tensor<8x16384x128xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,128,16384]{1,2,0}"} : (tensor<8x16384x128xf32>) -> tensor<8x128x16384xf32>
      %411 = mhlo.copy %410 : tensor<8x128x16384xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x128x16384xf32>) -> tensor<8x128x128x128xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{2,1,3,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %414 = mhlo.copy %413 : tensor<8x128x128x128xf32>
      mhlo.return %414 : tensor<8x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x128xf32>, tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %41 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x128xf32>
      mhlo.return %408 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<8x128x128x128xf32>
    %43 = "mhlo.fusion"(%42, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x128x128x128xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x128x128x128xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{1,3,2,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %412 = mhlo.copy %411 : tensor<8x128x128x128xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x128xf32>) -> tensor<8x128x16384xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,128]{1,2,0}"} : (tensor<8x128x16384xf32>) -> tensor<8x16384x128xf32>
      %415 = mhlo.copy %414 : tensor<8x16384x128xf32>
      %416 = stablehlo.negate %409 : tensor<8x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x128x128x128xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{1,3,2,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %420 = mhlo.copy %419 : tensor<8x128x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x128xf32>) -> tensor<8x128x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,128]{1,2,0}"} : (tensor<8x128x16384xf32>) -> tensor<8x16384x128xf32>
      %423 = mhlo.copy %422 : tensor<8x16384x128xf32>
      %424 = stablehlo.abs %423 : tensor<8x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x16384x128xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x16384x128xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x16384x128xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x16384x128xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x16384x128xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x16384x128xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x16384x128xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x16384x128xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %452 = stablehlo.negate %428 : tensor<8x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %455 = stablehlo.exponential %452 : tensor<8x16384x128xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x16384x128xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x16384x128xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x16384x128xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x16384x128xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x16384x128xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x16384x128xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x16384x128xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x16384x128xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x16384x128xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x16384x128xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x16384x128xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x16384x128xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x16384x128xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x16384x128xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x16384x128xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x16384x128xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x16384x128xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x16384x128xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x16384x128xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x16384x128xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x128xf32>) -> tensor<131072x128xf32>
      mhlo.return %514 : tensor<131072x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x128x128xf32>, tensor<128xf32>) -> tensor<131072x128xf32>
    %44 = stablehlo.dot_general %43, %arg26, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x128xf32>, tensor<32x128xf32>) -> tensor<131072x32xf32>
    %45 = "mhlo.fusion"(%35, %44, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<131072x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x16384x32xf32>
      mhlo.return %410 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<131072x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %46 = "mhlo.fusion"(%45, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %47 = "mhlo.fusion"(%45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.reduce(%411 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %413 = stablehlo.multiply %412, %407 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.add %413, %414 : tensor<8x16384xf32>
      %416 = stablehlo.rsqrt %415 : tensor<8x16384xf32>
      mhlo.return %416 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x16384xf32>
    %48 = "mhlo.fusion"(%arg28, %arg29, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
      mhlo.return %417 : tensor<131072x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<131072x32xf32>
    %49 = stablehlo.dot_general %48, %arg30, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<32x32xf32>) -> tensor<131072x32xf32>
    %50 = "mhlo.fusion"(%49, %arg31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      mhlo.return %409 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %51 = "mhlo.fusion"(%arg28, %arg29, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = stablehlo.transpose %416, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,32,16384]{1,2,0}"} : (tensor<8x16384x32xf32>) -> tensor<8x32x16384xf32>
      %418 = mhlo.copy %417 : tensor<8x32x16384xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x32x16384xf32>) -> tensor<8x32x128x128xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,32]{2,1,3,0}"} : (tensor<8x32x128x128xf32>) -> tensor<8x128x128x32xf32>
      %421 = mhlo.copy %420 : tensor<8x128x128x32xf32>
      mhlo.return %421 : tensor<8x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x128x128x32xf32>
    %52 = "mhlo.fusion"(%arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %408 = mhlo.copy %407 : tensor<8x8x32x32xf32>
      mhlo.return %408 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %53 = stablehlo.convolution(%51, %52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<8x16x16x32xf32>
    %54 = "mhlo.fusion"(%53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      mhlo.return %413 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<8x256x32xf32>
    %55 = "mhlo.fusion"(%54, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
    %56 = "mhlo.fusion"(%55, %53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x32xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %419 = stablehlo.reduce(%418 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x256x32xf32>, tensor<f32>) -> tensor<8x256xf32>
      %420 = stablehlo.multiply %419, %414 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<8x256xf32>
      %423 = stablehlo.rsqrt %422 : tensor<8x256xf32>
      mhlo.return %423 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<8x256xf32>
    %57 = "mhlo.fusion"(%arg34, %arg35, %56, %55, %53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x32xf32>, %arg213: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<8x16x16x32xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x32xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,16,16]{1,3,2,0}"} : (tensor<8x16x16x32xf32>) -> tensor<8x32x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x32x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x16x16xf32>) -> tensor<8x32x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,32]{1,2,0}"} : (tensor<8x32x256xf32>) -> tensor<8x256x32xf32>
      %413 = mhlo.copy %412 : tensor<8x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x32xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x32xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x256x32xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x32xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x256x32xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x32xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x32xf32>) -> tensor<2048x32xf32>
      mhlo.return %424 : tensor<2048x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x32xf32>, tensor<32xf32>) -> tensor<2048x32xf32>
    %58 = stablehlo.dot_general %57, %arg36, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x32xf32>, tensor<32x32xf32>) -> tensor<2048x32xf32>
    %59 = "mhlo.fusion"(%58, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2048x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[8,1,256,32]{3,2,0,1}"} : (tensor<2048x32xf32>) -> tensor<8x1x256x32xf32>
      %410 = mhlo.copy %409 : tensor<8x1x256x32xf32>
      %411 = stablehlo.transpose %410, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[8,1,32,256]{2,3,1,0}"} : (tensor<8x1x256x32xf32>) -> tensor<8x1x32x256xf32>
      %412 = mhlo.copy %411 : tensor<8x1x32x256xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x32x256xf32>) -> tensor<8x32x256xf32>
      mhlo.return %413 : tensor<8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x32xf32>, tensor<32xf32>) -> tensor<8x32x256xf32>
    %60 = stablehlo.dot_general %50, %59, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16384x32xf32>, tensor<8x32x256xf32>) -> tensor<8x16384x256xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x16384x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[8,1,16384,256]{3,2,0,1}"} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
      %410 = mhlo.copy %409 : tensor<8x1x16384x256xf32>
      mhlo.return %410 : tensor<8x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
    %62 = "mhlo.fusion"(%cst, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<8x1x16384x256xf32>):
      %407 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %409 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %409 : tensor<f32>
      }) : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384x8xf32>
      %408 = stablehlo.reduce(%407 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<8x1x16384x8xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
      mhlo.return %408 : tensor<8x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<8x1x16384x256xf32>) -> tensor<8x1x16384xf32>
    %63 = "mhlo.fusion"(%62, %60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1x16384xf32>, %arg209: tensor<8x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[8,1,16384,256]{3,2,0,1}"} : (tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
      %410 = mhlo.copy %409 : tensor<8x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1x16384xf32>
      %412 = stablehlo.maximum %411, %arg208 : tensor<8x1x16384xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x16384xf32>) -> tensor<8x16384xf32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [0, 2] : (tensor<8x16384xf32>) -> tensor<8x1x16384x256xf32>
      %415 = stablehlo.subtract %410, %414 : tensor<8x1x16384x256xf32>
      %416 = stablehlo.exponential %415 : tensor<8x1x16384x256xf32>
      mhlo.return %416 : tensor<8x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384xf32>, tensor<8x16384x256xf32>) -> tensor<8x1x16384x256xf32>
    %64 = "mhlo.fusion"(%63, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1x16384x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
      mhlo.return %407 : tensor<8x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384x256xf32>, tensor<f32>) -> tensor<8x1x16384xf32>
    %65 = "mhlo.fusion"(%63, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1x16384x256xf32>, %arg209: tensor<8x1x16384xf32>):
      %407 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1x16384xf32>) -> tensor<8x16384xf32>
      %408 = stablehlo.broadcast_in_dim %407, dims = [0, 2] : (tensor<8x16384xf32>) -> tensor<8x1x16384x256xf32>
      %409 = stablehlo.divide %arg208, %408 : tensor<8x1x16384x256xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x16384x256xf32>) -> tensor<8x16384x256xf32>
      mhlo.return %410 : tensor<8x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1x16384x256xf32>, tensor<8x1x16384xf32>) -> tensor<8x16384x256xf32>
    %66 = stablehlo.dot_general %57, %arg38, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x32xf32>, tensor<32x32xf32>) -> tensor<2048x32xf32>
    %67 = "mhlo.fusion"(%66, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x32xf32>, %arg209: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<2048x32xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %409 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x32xf32>, tensor<32xf32>) -> tensor<8x256x32xf32>
    %68 = stablehlo.dot_general %65, %67, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16384x256xf32>, tensor<8x256x32xf32>) -> tensor<8x16384x32xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
    %70 = stablehlo.dot_general %69, %arg40, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<32x32xf32>) -> tensor<131072x32xf32>
    %71 = "mhlo.fusion"(%45, %70, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<131072x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x16384x32xf32>
      mhlo.return %410 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<131072x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %72 = "mhlo.fusion"(%71, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.reduce(%411 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %413 = stablehlo.multiply %412, %407 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.add %413, %414 : tensor<8x16384xf32>
      %416 = stablehlo.rsqrt %415 : tensor<8x16384xf32>
      mhlo.return %416 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x16384xf32>
    %74 = "mhlo.fusion"(%arg42, %arg43, %73, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<131072x32xf32>
      mhlo.return %417 : tensor<131072x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<131072x32xf32>
    %75 = stablehlo.dot_general %74, %arg44, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x32xf32>, tensor<128x32xf32>) -> tensor<131072x128xf32>
    %76 = "mhlo.fusion"(%75, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<131072x128xf32>, %arg209: tensor<128xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<131072x128xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<131072x128xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x128xf32>) -> tensor<8x16384x128xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,128,16384]{1,2,0}"} : (tensor<8x16384x128xf32>) -> tensor<8x128x16384xf32>
      %411 = mhlo.copy %410 : tensor<8x128x16384xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x128x16384xf32>) -> tensor<8x128x128x128xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{2,1,3,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %414 = mhlo.copy %413 : tensor<8x128x128x128xf32>
      mhlo.return %414 : tensor<8x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<131072x128xf32>, tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %77 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x128xf32>
      mhlo.return %408 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %78 = stablehlo.convolution(%76, %77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<8x128x128x128xf32>
    %79 = "mhlo.fusion"(%78, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x128x128x128xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x128x128x128xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{1,3,2,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %412 = mhlo.copy %411 : tensor<8x128x128x128xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x128xf32>) -> tensor<8x128x16384xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,128]{1,2,0}"} : (tensor<8x128x16384xf32>) -> tensor<8x16384x128xf32>
      %415 = mhlo.copy %414 : tensor<8x16384x128xf32>
      %416 = stablehlo.negate %409 : tensor<8x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x128x128x128xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,128]{1,3,2,0}"} : (tensor<8x128x128x128xf32>) -> tensor<8x128x128x128xf32>
      %420 = mhlo.copy %419 : tensor<8x128x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x128xf32>) -> tensor<8x128x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,16384,128]{1,2,0}"} : (tensor<8x128x16384xf32>) -> tensor<8x16384x128xf32>
      %423 = mhlo.copy %422 : tensor<8x16384x128xf32>
      %424 = stablehlo.abs %423 : tensor<8x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x16384x128xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x16384x128xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x16384x128xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x16384x128xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x16384x128xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x16384x128xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x16384x128xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x16384x128xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %452 = stablehlo.negate %428 : tensor<8x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %455 = stablehlo.exponential %452 : tensor<8x16384x128xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x16384x128xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x16384x128xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x16384x128xf32>, tensor<8x16384x128xf32>) -> tensor<8x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x16384x128xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x16384x128xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x16384x128xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x16384x128xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x16384x128xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x16384x128xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x16384x128xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x16384x128xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x16384x128xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x16384x128xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x16384x128xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x16384x128xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x16384x128xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x16384x128xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x16384x128xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x16384x128xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x16384x128xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x16384x128xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x16384x128xi1>, tensor<8x16384x128xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x16384x128xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x128xf32>) -> tensor<131072x128xf32>
      mhlo.return %514 : tensor<131072x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x128x128xf32>, tensor<128xf32>) -> tensor<131072x128xf32>
    %80 = stablehlo.dot_general %79, %arg48, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131072x128xf32>, tensor<32x128xf32>) -> tensor<131072x32xf32>
    %81 = "mhlo.fusion"(%71, %80, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<131072x32xf32>, %arg210: tensor<32xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<131072x32xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<131072x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x32xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x16384x32xf32>
      mhlo.return %410 : tensor<8x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<131072x32xf32>, tensor<32xf32>) -> tensor<8x16384x32xf32>
    %82 = "mhlo.fusion"(%81, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      mhlo.return %407 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
    %83 = "mhlo.fusion"(%81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16384x32xf32>, %arg209: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.reduce(%411 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<8x16384x32xf32>, tensor<f32>) -> tensor<8x16384xf32>
      %413 = stablehlo.multiply %412, %407 : tensor<8x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %415 = stablehlo.add %413, %414 : tensor<8x16384xf32>
      %416 = stablehlo.rsqrt %415 : tensor<8x16384xf32>
      mhlo.return %416 : tensor<8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x16384xf32>
    %84 = "mhlo.fusion"(%arg50, %arg51, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<8x128x128x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,128,128]{1,3,2,0}"} : (tensor<8x128x128x32xf32>) -> tensor<8x32x128x128xf32>
      %419 = mhlo.copy %418 : tensor<8x32x128x128xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x32x128x128xf32>) -> tensor<8x32x16384xf32>
      mhlo.return %420 : tensor<8x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x32x16384xf32>
    %85 = "mhlo.fusion"(%arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>):
      %407 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %408 = mhlo.copy %407 : tensor<32x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<8x32x256xf32>
      mhlo.return %409 : tensor<8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>) -> tensor<8x32x256xf32>
    %86 = stablehlo.dot_general %84, %85, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x32x16384xf32>, tensor<8x32x256xf32>) -> tensor<8x16384x256xf32>
    %87 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<63> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %409 = stablehlo.iota dim = 0 : tensor<128xi32>
      %410 = stablehlo.convert %409 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %412 = stablehlo.add %410, %411 : tensor<128xf32>
      %413 = stablehlo.multiply %412, %411 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.add %413, %414 : tensor<128xf32>
      %416 = stablehlo.maximum %408, %415 : tensor<128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %418 = stablehlo.convert %417 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<1> : tensor<i32>
      %419 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %420 = stablehlo.add %418, %419 : tensor<128x1xi32>
      %421 = stablehlo.minimum %407, %420 : tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<0> : tensor<i32>
      %422 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %423 = stablehlo.compare LT, %421, %422 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<64> : tensor<i32>
      %424 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %425 = stablehlo.add %421, %424 : tensor<128x1xi32>
      %426 = stablehlo.select %423, %425, %421 : tensor<128x1xi1>, tensor<128x1xi32>
      %427 = mhlo.bitcast %426 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %428 = stablehlo.broadcast_in_dim %427, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %429 = stablehlo.convert %416 : (tensor<128xf32>) -> tensor<128xi32>
      %430 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %432 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %433 = stablehlo.add %429, %432 : tensor<128xi32>
      %434 = stablehlo.select %431, %433, %429 : tensor<128xi1>, tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %436 = stablehlo.concatenate %428, %435, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %436 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %88 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %408 = stablehlo.iota dim = 0 : tensor<128xi32>
      %409 = stablehlo.convert %408 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %411 = stablehlo.add %409, %410 : tensor<128xf32>
      %412 = stablehlo.multiply %411, %410 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.add %412, %413 : tensor<128xf32>
      %415 = stablehlo.maximum %407, %414 : tensor<128xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %417 = stablehlo.convert %416 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %418 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %419 = stablehlo.compare LT, %417, %418 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<64> : tensor<i32>
      %420 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %421 = stablehlo.add %417, %420 : tensor<128x1xi32>
      %422 = stablehlo.select %419, %421, %417 : tensor<128x1xi1>, tensor<128x1xi32>
      %423 = mhlo.bitcast %422 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %425 = stablehlo.convert %415 : (tensor<128xf32>) -> tensor<128xi32>
      %426 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %427 = stablehlo.compare LT, %425, %426 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %428 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %429 = stablehlo.add %425, %428 : tensor<128xi32>
      %430 = stablehlo.select %427, %429, %425 : tensor<128xi1>, tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %432 = stablehlo.concatenate %424, %431, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %432 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %89 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %407 = stablehlo.iota dim = 0 : tensor<128xi32>
      %408 = stablehlo.convert %407 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %410 = stablehlo.add %408, %409 : tensor<128xf32>
      mhlo.return %410 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %90 = "mhlo.fusion"(%arg50, %arg51, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<8x16384xf32>, %arg211: tensor<8x16384x32xf32>, %arg212: tensor<8x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16384xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x16384xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x16384xf32>) -> tensor<8x16384x32xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x16384x32xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x16384x32xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<8x16384x32xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x16384x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16384x32xf32>) -> tensor<8x128x128x32xf32>
      mhlo.return %417 : tensor<8x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<8x16384xf32>, tensor<8x16384x32xf32>, tensor<8x16384xf32>) -> tensor<8x128x128x32xf32>
    %91 = "mhlo.fusion"(%arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %408 = mhlo.copy %407 : tensor<3x3x32x64xf32>
      mhlo.return %408 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %92 = stablehlo.convolution(%90, %91) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x128x128x32xf32>, tensor<3x3x32x64xf32>) -> tensor<8x64x64x64xf32>
    %93 = "mhlo.fusion"(%92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x64x64x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      mhlo.return %413 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %94 = "mhlo.fusion"(%93, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %95 = "mhlo.fusion"(%94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>, %arg209: tensor<8x64x64x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x4096xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x4096x64xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x4096x64xf32>
      mhlo.return %418 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %96 = "mhlo.fusion"(%95, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %97 = "mhlo.fusion"(%96, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %98 = "mhlo.fusion"(%97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %99 = "mhlo.fusion"(%arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096xf32>, %arg212: tensor<8x64x64x64xf32>, %arg213: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x4096xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x4096x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x4096x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x4096x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x4096x64xf32>
      mhlo.return %423 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %100 = "mhlo.fusion"(%99, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %101 = "mhlo.fusion"(%100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<8x4096xf32>, %arg212: tensor<8x4096xf32>, %arg213: tensor<8x64x64x64xf32>, %arg214: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg213, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %415 = stablehlo.multiply %arg212, %414 : tensor<8x4096xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x4096x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x4096x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x4096x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x4096x64xf32>
      %424 = stablehlo.multiply %arg208, %414 : tensor<8x4096xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x4096x64xf32>
      %427 = stablehlo.multiply %426, %426 : tensor<8x4096x64xf32>
      mhlo.return %427 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %102 = "mhlo.fusion"(%101, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %103 = "mhlo.fusion"(%102, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %105 = "mhlo.fusion"(%arg57, %arg58, %104, %100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<8x4096xf32>, %arg215: tensor<8x4096xf32>, %arg216: tensor<8x64x64x64xf32>, %arg217: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x4096xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x4096x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x4096x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x4096x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x4096x64xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x4096xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x4096x64xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x4096x64xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x4096x64xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x4096x64xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %433 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<32768x64xf32>
    %106 = stablehlo.dot_general %105, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x64xf32>) -> tensor<32768x64xf32>
    %107 = "mhlo.fusion"(%106, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,4096,32]{3,1,2,0}"} : (tensor<8x4096x2x32xf32>) -> tensor<8x2x4096x32xf32>
      %411 = mhlo.copy %410 : tensor<8x2x4096x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x4096x32xf32>) -> tensor<16x4096x32xf32>
      mhlo.return %412 : tensor<16x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>, tensor<64xf32>) -> tensor<16x4096x32xf32>
    %108 = "mhlo.fusion"(%arg57, %arg58, %104, %100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<8x4096xf32>, %arg215: tensor<8x4096xf32>, %arg216: tensor<8x64x64x64xf32>, %arg217: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x64x64x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %410 = mhlo.copy %409 : tensor<8x64x64x64xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %413 = mhlo.copy %412 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x4096xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x4096x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x4096x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x4096x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x4096x64xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x4096xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x4096x64xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x4096x64xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x4096x64xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x4096x64xf32>
      %433 = stablehlo.transpose %432, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,64,4096]{1,2,0}"} : (tensor<8x4096x64xf32>) -> tensor<8x64x4096xf32>
      %434 = mhlo.copy %433 : tensor<8x64x4096xf32>
      %435 = mhlo.bitcast %434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x64x4096xf32>) -> tensor<8x64x64x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{2,1,3,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %437 = mhlo.copy %436 : tensor<8x64x64x64xf32>
      mhlo.return %437 : tensor<8x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>) -> tensor<8x64x64x64xf32>
    %109 = "mhlo.fusion"(%arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %408 = mhlo.copy %407 : tensor<4x4x64x64xf32>
      mhlo.return %408 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %110 = stablehlo.convolution(%108, %109) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<8x16x16x64xf32>
    %111 = "mhlo.fusion"(%110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      mhlo.return %413 : tensor<8x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<8x256x64xf32>
    %112 = "mhlo.fusion"(%111, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256xf32>
    %113 = "mhlo.fusion"(%112, %110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x64xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x64xf32>
      mhlo.return %418 : tensor<8x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<8x256x64xf32>
    %114 = "mhlo.fusion"(%113, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256x2xf32>
      mhlo.return %407 : tensor<8x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256x2xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x2xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x2xf32>) -> tensor<8x256xf32>
    %116 = "mhlo.fusion"(%arg63, %arg64, %115, %112, %110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x64xf32>, %arg213: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x256x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x256x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x64xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x64xf32>) -> tensor<2048x64xf32>
      mhlo.return %424 : tensor<2048x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<2048x64xf32>
    %117 = stablehlo.dot_general %116, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %118 = "mhlo.fusion"(%117, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<2048x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x64xf32>) -> tensor<8x256x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,32,256]{2,1,3,0}"} : (tensor<8x256x2x32xf32>) -> tensor<8x2x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x2x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x32x256xf32>) -> tensor<16x32x256xf32>
      mhlo.return %412 : tensor<16x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x64xf32>, tensor<64xf32>) -> tensor<16x32x256xf32>
    %119 = stablehlo.dot_general %107, %118, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x4096x32xf32>, tensor<16x32x256xf32>) -> tensor<16x4096x256xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<16x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
      mhlo.return %409 : tensor<8x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
    %121 = "mhlo.fusion"(%120, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096x8xf32>
      mhlo.return %407 : tensor<8x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096x8xf32>
    %122 = "mhlo.fusion"(%121, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x2x4096x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x2x4096x8xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
      mhlo.return %407 : tensor<8x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x8xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
    %123 = "mhlo.fusion"(%122, %119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096xf32>, %arg209: tensor<16x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<16x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x2x4096xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x2x4096xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x2x4096xf32>) -> tensor<8x2x4096x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x2x4096x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x2x4096x256xf32>
      mhlo.return %414 : tensor<8x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096xf32>, tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
    %124 = "mhlo.fusion"(%123, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
      mhlo.return %407 : tensor<8x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
    %125 = "mhlo.fusion"(%123, %124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<8x2x4096xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x2x4096xf32>) -> tensor<8x2x4096x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x2x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x4096x256xf32>) -> tensor<16x4096x256xf32>
      mhlo.return %409 : tensor<16x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<8x2x4096xf32>) -> tensor<16x4096x256xf32>
    %126 = stablehlo.dot_general %116, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %127 = "mhlo.fusion"(%126, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<2048x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x64xf32>) -> tensor<8x256x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,256,32]{3,1,2,0}"} : (tensor<8x256x2x32xf32>) -> tensor<8x2x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x2x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x256x32xf32>) -> tensor<16x256x32xf32>
      mhlo.return %412 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x64xf32>, tensor<64xf32>) -> tensor<16x256x32xf32>
    %128 = stablehlo.dot_general %125, %127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x4096x256xf32>, tensor<16x256x32xf32>) -> tensor<16x4096x32xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x32xf32>) -> tensor<8x2x4096x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,4096,2,32]{3,1,2,0}"} : (tensor<8x2x4096x32xf32>) -> tensor<8x4096x2x32xf32>
      %409 = mhlo.copy %408 : tensor<8x4096x2x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x4096x2x32xf32>) -> tensor<32768x64xf32>
      mhlo.return %410 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x32xf32>) -> tensor<32768x64xf32>
    %130 = stablehlo.dot_general %129, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x64xf32>) -> tensor<32768x64xf32>
    %131 = "mhlo.fusion"(%130, %arg55, %arg56, %98, %94, %92, %arg54, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<8x4096xf32>, %arg212: tensor<8x4096xf32>, %arg213: tensor<8x64x64x64xf32>, %arg214: tensor<64xf32>, %arg215: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<8x64x64x64xf32>
      %411 = stablehlo.add %arg213, %410 : tensor<8x64x64x64xf32>
      %412 = stablehlo.transpose %411, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %413 = mhlo.copy %412 : tensor<8x64x64x64xf32>
      %414 = mhlo.bitcast %413 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      %415 = stablehlo.transpose %414, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,64]{1,2,0}"} : (tensor<8x64x4096xf32>) -> tensor<8x4096x64xf32>
      %416 = mhlo.copy %415 : tensor<8x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<8x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %420 = stablehlo.subtract %416, %419 : tensor<8x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<8x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<8x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<8x4096x64xf32>
      %427 = stablehlo.add %409, %426 : tensor<8x4096x64xf32>
      mhlo.return %427 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096xf32>, tensor<8x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %132 = "mhlo.fusion"(%131, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %133 = "mhlo.fusion"(%131, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x4096x64xf32>
      mhlo.return %411 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
    %134 = "mhlo.fusion"(%133, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %135 = "mhlo.fusion"(%134, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %136 = "mhlo.fusion"(%135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %137 = "mhlo.fusion"(%arg71, %arg72, %136, %131, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %417 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<32768x64xf32>
    %138 = stablehlo.dot_general %137, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<256x64xf32>) -> tensor<32768x256xf32>
    %139 = "mhlo.fusion"(%138, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<32768x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<8x4096x256xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,4096]{1,2,0}"} : (tensor<8x4096x256xf32>) -> tensor<8x256x4096xf32>
      %411 = mhlo.copy %410 : tensor<8x256x4096xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x4096xf32>) -> tensor<8x256x64x64xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,256]{2,1,3,0}"} : (tensor<8x256x64x64xf32>) -> tensor<8x64x64x256xf32>
      %414 = mhlo.copy %413 : tensor<8x64x64x256xf32>
      mhlo.return %414 : tensor<8x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<8x64x64x256xf32>
    %140 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x256xf32>
      mhlo.return %408 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %141 = stablehlo.convolution(%139, %140) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<8x64x64x256xf32>
    %142 = "mhlo.fusion"(%141, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x64x64x256xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<8x64x64x256xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x64x64x256xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x64x64x256xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,64,64]{1,3,2,0}"} : (tensor<8x64x64x256xf32>) -> tensor<8x256x64x64xf32>
      %412 = mhlo.copy %411 : tensor<8x256x64x64xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x64x64xf32>) -> tensor<8x256x4096xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,256]{1,2,0}"} : (tensor<8x256x4096xf32>) -> tensor<8x4096x256xf32>
      %415 = mhlo.copy %414 : tensor<8x4096x256xf32>
      %416 = stablehlo.negate %409 : tensor<8x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x64x64x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x64x64x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,64,64]{1,3,2,0}"} : (tensor<8x64x64x256xf32>) -> tensor<8x256x64x64xf32>
      %420 = mhlo.copy %419 : tensor<8x256x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x64x64xf32>) -> tensor<8x256x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,256]{1,2,0}"} : (tensor<8x256x4096xf32>) -> tensor<8x4096x256xf32>
      %423 = mhlo.copy %422 : tensor<8x4096x256xf32>
      %424 = stablehlo.abs %423 : tensor<8x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x4096x256xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x4096x256xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x4096x256xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x4096x256xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x4096x256xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x4096x256xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x4096x256xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x4096x256xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %452 = stablehlo.negate %428 : tensor<8x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %455 = stablehlo.exponential %452 : tensor<8x4096x256xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x4096x256xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x4096x256xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x4096x256xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x4096x256xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x4096x256xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x4096x256xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x4096x256xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x4096x256xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x4096x256xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x4096x256xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x4096x256xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x4096x256xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x4096x256xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x4096x256xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x4096x256xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x4096x256xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x4096x256xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x4096x256xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x4096x256xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x4096x256xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %514 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x64x64x256xf32>, tensor<256xf32>) -> tensor<32768x256xf32>
    %143 = stablehlo.dot_general %142, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<64x256xf32>) -> tensor<32768x64xf32>
    %144 = "mhlo.fusion"(%131, %143, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<32768x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x4096x64xf32>
      mhlo.return %410 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<32768x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %145 = "mhlo.fusion"(%144, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %146 = "mhlo.fusion"(%144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x4096x64xf32>
      mhlo.return %411 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
    %147 = "mhlo.fusion"(%146, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %148 = "mhlo.fusion"(%147, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %150 = "mhlo.fusion"(%arg79, %arg80, %149, %144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %417 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<32768x64xf32>
    %151 = stablehlo.dot_general %150, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x64xf32>) -> tensor<32768x64xf32>
    %152 = "mhlo.fusion"(%151, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,4096,32]{3,1,2,0}"} : (tensor<8x4096x2x32xf32>) -> tensor<8x2x4096x32xf32>
      %411 = mhlo.copy %410 : tensor<8x2x4096x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x4096x32xf32>) -> tensor<16x4096x32xf32>
      mhlo.return %412 : tensor<16x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x64xf32>, tensor<64xf32>) -> tensor<16x4096x32xf32>
    %153 = "mhlo.fusion"(%arg79, %arg80, %149, %144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = stablehlo.transpose %416, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,64,4096]{1,2,0}"} : (tensor<8x4096x64xf32>) -> tensor<8x64x4096xf32>
      %418 = mhlo.copy %417 : tensor<8x64x4096xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x64x4096xf32>) -> tensor<8x64x64x64xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{2,1,3,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %421 = mhlo.copy %420 : tensor<8x64x64x64xf32>
      mhlo.return %421 : tensor<8x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x64x64x64xf32>
    %154 = "mhlo.fusion"(%arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %408 = mhlo.copy %407 : tensor<4x4x64x64xf32>
      mhlo.return %408 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %155 = stablehlo.convolution(%153, %154) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<8x16x16x64xf32>
    %156 = "mhlo.fusion"(%155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      mhlo.return %413 : tensor<8x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<8x256x64xf32>
    %157 = "mhlo.fusion"(%156, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256xf32>
    %158 = "mhlo.fusion"(%157, %155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x64xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x64xf32>
      mhlo.return %418 : tensor<8x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<8x256x64xf32>
    %159 = "mhlo.fusion"(%158, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256x2xf32>
      mhlo.return %407 : tensor<8x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x64xf32>, tensor<f32>) -> tensor<8x256x2xf32>
    %160 = "mhlo.fusion"(%159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x2xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x2xf32>) -> tensor<8x256xf32>
    %161 = "mhlo.fusion"(%arg85, %arg86, %160, %157, %155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x64xf32>, %arg213: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<8x16x16x64xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x64xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,16,16]{1,3,2,0}"} : (tensor<8x16x16x64xf32>) -> tensor<8x64x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x64x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x16x16xf32>) -> tensor<8x64x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,64]{1,2,0}"} : (tensor<8x64x256xf32>) -> tensor<8x256x64xf32>
      %413 = mhlo.copy %412 : tensor<8x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x64xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x64xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x256x64xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x64xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x256x64xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x64xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x64xf32>) -> tensor<2048x64xf32>
      mhlo.return %424 : tensor<2048x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x64xf32>, tensor<64xf32>) -> tensor<2048x64xf32>
    %162 = stablehlo.dot_general %161, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %163 = "mhlo.fusion"(%162, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<2048x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x64xf32>) -> tensor<8x256x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,32,256]{2,1,3,0}"} : (tensor<8x256x2x32xf32>) -> tensor<8x2x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x2x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x32x256xf32>) -> tensor<16x32x256xf32>
      mhlo.return %412 : tensor<16x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x64xf32>, tensor<64xf32>) -> tensor<16x32x256xf32>
    %164 = stablehlo.dot_general %152, %163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x4096x32xf32>, tensor<16x32x256xf32>) -> tensor<16x4096x256xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<16x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
      mhlo.return %409 : tensor<8x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
    %166 = "mhlo.fusion"(%165, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096x8xf32>
      mhlo.return %407 : tensor<8x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096x8xf32>
    %167 = "mhlo.fusion"(%166, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x2x4096x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x2x4096x8xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
      mhlo.return %407 : tensor<8x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x8xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
    %168 = "mhlo.fusion"(%167, %164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096xf32>, %arg209: tensor<16x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<16x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x2x4096xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x2x4096xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x2x4096xf32>) -> tensor<8x2x4096x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x2x4096x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x2x4096x256xf32>
      mhlo.return %414 : tensor<8x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096xf32>, tensor<16x4096x256xf32>) -> tensor<8x2x4096x256xf32>
    %169 = "mhlo.fusion"(%168, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
      mhlo.return %407 : tensor<8x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<f32>) -> tensor<8x2x4096xf32>
    %170 = "mhlo.fusion"(%168, %169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x2x4096x256xf32>, %arg209: tensor<8x2x4096xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x2x4096xf32>) -> tensor<8x2x4096x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x2x4096x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x4096x256xf32>) -> tensor<16x4096x256xf32>
      mhlo.return %409 : tensor<16x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2x4096x256xf32>, tensor<8x2x4096xf32>) -> tensor<16x4096x256xf32>
    %171 = stablehlo.dot_general %161, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x64xf32>, tensor<64x64xf32>) -> tensor<2048x64xf32>
    %172 = "mhlo.fusion"(%171, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x64xf32>, %arg209: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<2048x64xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x64xf32>) -> tensor<8x256x2x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,2,256,32]{3,1,2,0}"} : (tensor<8x256x2x32xf32>) -> tensor<8x2x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x2x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x2x256x32xf32>) -> tensor<16x256x32xf32>
      mhlo.return %412 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x64xf32>, tensor<64xf32>) -> tensor<16x256x32xf32>
    %173 = stablehlo.dot_general %170, %172, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x4096x256xf32>, tensor<16x256x32xf32>) -> tensor<16x4096x32xf32>
    %174 = "mhlo.fusion"(%173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x32xf32>) -> tensor<8x2x4096x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,4096,2,32]{3,1,2,0}"} : (tensor<8x2x4096x32xf32>) -> tensor<8x4096x2x32xf32>
      %409 = mhlo.copy %408 : tensor<8x4096x2x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x4096x2x32xf32>) -> tensor<32768x64xf32>
      mhlo.return %410 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x32xf32>) -> tensor<32768x64xf32>
    %175 = stablehlo.dot_general %174, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<64x64xf32>) -> tensor<32768x64xf32>
    %176 = "mhlo.fusion"(%144, %175, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<32768x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x4096x64xf32>
      mhlo.return %410 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<32768x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %177 = "mhlo.fusion"(%176, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %178 = "mhlo.fusion"(%176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x4096x64xf32>
      mhlo.return %411 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
    %179 = "mhlo.fusion"(%178, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %180 = "mhlo.fusion"(%179, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %181 = "mhlo.fusion"(%180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %182 = "mhlo.fusion"(%arg93, %arg94, %181, %176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<32768x64xf32>
      mhlo.return %417 : tensor<32768x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<32768x64xf32>
    %183 = stablehlo.dot_general %182, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x64xf32>, tensor<256x64xf32>) -> tensor<32768x256xf32>
    %184 = "mhlo.fusion"(%183, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32768x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<32768x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<32768x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<8x4096x256xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,4096]{1,2,0}"} : (tensor<8x4096x256xf32>) -> tensor<8x256x4096xf32>
      %411 = mhlo.copy %410 : tensor<8x256x4096xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x4096xf32>) -> tensor<8x256x64x64xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,256]{2,1,3,0}"} : (tensor<8x256x64x64xf32>) -> tensor<8x64x64x256xf32>
      %414 = mhlo.copy %413 : tensor<8x64x64x256xf32>
      mhlo.return %414 : tensor<8x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32768x256xf32>, tensor<256xf32>) -> tensor<8x64x64x256xf32>
    %185 = "mhlo.fusion"(%arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x256xf32>
      mhlo.return %408 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %186 = stablehlo.convolution(%184, %185) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<8x64x64x256xf32>
    %187 = "mhlo.fusion"(%186, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x64x64x256xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<8x64x64x256xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x64x64x256xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x64x64x256xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,64,64]{1,3,2,0}"} : (tensor<8x64x64x256xf32>) -> tensor<8x256x64x64xf32>
      %412 = mhlo.copy %411 : tensor<8x256x64x64xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x64x64xf32>) -> tensor<8x256x4096xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,256]{1,2,0}"} : (tensor<8x256x4096xf32>) -> tensor<8x4096x256xf32>
      %415 = mhlo.copy %414 : tensor<8x4096x256xf32>
      %416 = stablehlo.negate %409 : tensor<8x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x64x64x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x64x64x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,64,64]{1,3,2,0}"} : (tensor<8x64x64x256xf32>) -> tensor<8x256x64x64xf32>
      %420 = mhlo.copy %419 : tensor<8x256x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x64x64xf32>) -> tensor<8x256x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,4096,256]{1,2,0}"} : (tensor<8x256x4096xf32>) -> tensor<8x4096x256xf32>
      %423 = mhlo.copy %422 : tensor<8x4096x256xf32>
      %424 = stablehlo.abs %423 : tensor<8x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x4096x256xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x4096x256xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x4096x256xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x4096x256xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x4096x256xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x4096x256xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x4096x256xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x4096x256xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %452 = stablehlo.negate %428 : tensor<8x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %455 = stablehlo.exponential %452 : tensor<8x4096x256xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x4096x256xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x4096x256xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x4096x256xf32>, tensor<8x4096x256xf32>) -> tensor<8x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x4096x256xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x4096x256xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x4096x256xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x4096x256xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x4096x256xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x4096x256xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x4096x256xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x4096x256xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x4096x256xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x4096x256xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x4096x256xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x4096x256xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x4096x256xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x4096x256xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x4096x256xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x4096x256xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x4096x256xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x4096x256xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x4096x256xi1>, tensor<8x4096x256xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x4096x256xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x256xf32>) -> tensor<32768x256xf32>
      mhlo.return %514 : tensor<32768x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x64x64x256xf32>, tensor<256xf32>) -> tensor<32768x256xf32>
    %188 = stablehlo.dot_general %187, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<64x256xf32>) -> tensor<32768x64xf32>
    %189 = "mhlo.fusion"(%176, %188, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<32768x64xf32>, %arg210: tensor<64xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<32768x64xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<32768x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x64xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x4096x64xf32>
      mhlo.return %410 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<32768x64xf32>, tensor<64xf32>) -> tensor<8x4096x64xf32>
    %190 = "mhlo.fusion"(%189, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %191 = "mhlo.fusion"(%189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x4096x64xf32>
      mhlo.return %411 : tensor<8x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
    %192 = "mhlo.fusion"(%191, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096x64xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
      mhlo.return %407 : tensor<8x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x64xf32>, tensor<f32>) -> tensor<8x4096x2xf32>
    %193 = "mhlo.fusion"(%192, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x4096x2xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
      mhlo.return %407 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096x2xf32>, tensor<f32>) -> tensor<8x4096xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x4096xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x4096xf32>
      mhlo.return %411 : tensor<8x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %195 = "mhlo.fusion"(%arg101, %arg102, %194, %189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<8x64x64x64xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,64,64,64]{1,3,2,0}"} : (tensor<8x64x64x64xf32>) -> tensor<8x64x64x64xf32>
      %419 = mhlo.copy %418 : tensor<8x64x64x64xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x64x64x64xf32>) -> tensor<8x64x4096xf32>
      mhlo.return %420 : tensor<8x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x64x4096xf32>
    %196 = "mhlo.fusion"(%arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %407 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[64,256]{0,1}"} : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %408 = mhlo.copy %407 : tensor<64x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<8x64x256xf32>
      mhlo.return %409 : tensor<8x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<8x64x256xf32>
    %197 = stablehlo.dot_general %195, %196, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x64x4096xf32>, tensor<8x64x256xf32>) -> tensor<8x4096x256xf32>
    %198 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<31> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %409 = stablehlo.iota dim = 0 : tensor<128xi32>
      %410 = stablehlo.convert %409 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %412 = stablehlo.add %410, %411 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %416 = stablehlo.add %414, %415 : tensor<128xf32>
      %417 = stablehlo.maximum %408, %416 : tensor<128xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %419 = stablehlo.convert %418 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %420 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %421 = stablehlo.add %419, %420 : tensor<128x1xi32>
      %422 = stablehlo.minimum %407, %421 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %423 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %424 = stablehlo.compare LT, %422, %423 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<32> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %422, %425 : tensor<128x1xi32>
      %427 = stablehlo.select %424, %426, %422 : tensor<128x1xi1>, tensor<128x1xi32>
      %428 = mhlo.bitcast %427 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %430 = stablehlo.convert %417 : (tensor<128xf32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %433 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.add %430, %433 : tensor<128xi32>
      %435 = stablehlo.select %432, %434, %430 : tensor<128xi1>, tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %437 = stablehlo.concatenate %429, %436, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %437 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %199 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %408 = stablehlo.iota dim = 0 : tensor<128xi32>
      %409 = stablehlo.convert %408 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %411 = stablehlo.add %409, %410 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.add %413, %414 : tensor<128xf32>
      %416 = stablehlo.maximum %407, %415 : tensor<128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %418 = stablehlo.convert %417 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %419 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %420 = stablehlo.compare LT, %418, %419 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %421 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %422 = stablehlo.add %418, %421 : tensor<128x1xi32>
      %423 = stablehlo.select %420, %422, %418 : tensor<128x1xi1>, tensor<128x1xi32>
      %424 = mhlo.bitcast %423 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %426 = stablehlo.convert %416 : (tensor<128xf32>) -> tensor<128xi32>
      %427 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %428 = stablehlo.compare LT, %426, %427 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %429 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %430 = stablehlo.add %426, %429 : tensor<128xi32>
      %431 = stablehlo.select %428, %430, %426 : tensor<128xi1>, tensor<128xi32>
      %432 = stablehlo.broadcast_in_dim %431, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %433 = stablehlo.concatenate %425, %432, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %433 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %200 = "mhlo.fusion"(%arg101, %arg102, %194, %189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<8x4096xf32>, %arg211: tensor<8x4096x64xf32>, %arg212: tensor<8x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4096xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x4096xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x4096xf32>) -> tensor<8x4096x64xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x4096x64xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x4096x64xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<8x4096x64xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x4096x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x4096x64xf32>) -> tensor<8x64x64x64xf32>
      mhlo.return %417 : tensor<8x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<8x4096xf32>, tensor<8x4096x64xf32>, tensor<8x4096xf32>) -> tensor<8x64x64x64xf32>
    %201 = "mhlo.fusion"(%arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x64x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,160]{1,0,2,3}"} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
      %408 = mhlo.copy %407 : tensor<3x3x64x160xf32>
      mhlo.return %408 : tensor<3x3x64x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
    %202 = stablehlo.convolution(%200, %201) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x64x64x64xf32>, tensor<3x3x64x160xf32>) -> tensor<8x32x32x160xf32>
    %203 = "mhlo.fusion"(%202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x32x32x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      mhlo.return %413 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %204 = "mhlo.fusion"(%203, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %205 = "mhlo.fusion"(%204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>, %arg209: tensor<8x32x32x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x1024xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x1024x160xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x1024x160xf32>
      mhlo.return %418 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %206 = "mhlo.fusion"(%205, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %207 = "mhlo.fusion"(%206, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %209 = "mhlo.fusion"(%arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024xf32>, %arg212: tensor<8x32x32x160xf32>, %arg213: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x1024xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x1024x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x1024x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x1024x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x1024x160xf32>
      mhlo.return %423 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %210 = "mhlo.fusion"(%209, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %211 = "mhlo.fusion"(%210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<8x1024xf32>, %arg212: tensor<8x1024xf32>, %arg213: tensor<8x32x32x160xf32>, %arg214: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg213, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %415 = stablehlo.multiply %arg212, %414 : tensor<8x1024xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x1024x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x1024x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x1024x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x1024x160xf32>
      %424 = stablehlo.multiply %arg208, %414 : tensor<8x1024xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x1024x160xf32>
      %427 = stablehlo.multiply %426, %426 : tensor<8x1024x160xf32>
      mhlo.return %427 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %212 = "mhlo.fusion"(%211, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %213 = "mhlo.fusion"(%212, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %214 = "mhlo.fusion"(%213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %215 = "mhlo.fusion"(%arg108, %arg109, %214, %210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<8x1024xf32>, %arg215: tensor<8x1024xf32>, %arg216: tensor<8x32x32x160xf32>, %arg217: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x1024xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x1024x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x1024x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x1024x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x1024x160xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x1024xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x1024x160xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x1024x160xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x1024x160xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x1024x160xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8192x160xf32>
      mhlo.return %433 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8192x160xf32>
    %216 = stablehlo.dot_general %215, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<160x160xf32>) -> tensor<8192x160xf32>
    %217 = "mhlo.fusion"(%216, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8192x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,1024,32]{3,1,2,0}"} : (tensor<8x1024x5x32xf32>) -> tensor<8x5x1024x32xf32>
      %411 = mhlo.copy %410 : tensor<8x5x1024x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x1024x32xf32>) -> tensor<40x1024x32xf32>
      mhlo.return %412 : tensor<40x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8192x160xf32>, tensor<160xf32>) -> tensor<40x1024x32xf32>
    %218 = "mhlo.fusion"(%arg108, %arg109, %214, %210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<8x1024xf32>, %arg215: tensor<8x1024xf32>, %arg216: tensor<8x32x32x160xf32>, %arg217: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x32x32x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %410 = mhlo.copy %409 : tensor<8x160x32x32xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %413 = mhlo.copy %412 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x1024xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x1024x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x1024x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x1024x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x1024x160xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x1024xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x1024x160xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x1024x160xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x1024x160xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x1024x160xf32>
      %433 = stablehlo.transpose %432, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,160,1024]{1,2,0}"} : (tensor<8x1024x160xf32>) -> tensor<8x160x1024xf32>
      %434 = mhlo.copy %433 : tensor<8x160x1024xf32>
      %435 = mhlo.bitcast %434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x160x1024xf32>) -> tensor<8x160x32x32xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,32,160]{2,1,3,0}"} : (tensor<8x160x32x32xf32>) -> tensor<8x32x32x160xf32>
      %437 = mhlo.copy %436 : tensor<8x32x32x160xf32>
      mhlo.return %437 : tensor<8x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>) -> tensor<8x32x32x160xf32>
    %219 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %408 = mhlo.copy %407 : tensor<2x2x160x160xf32>
      mhlo.return %408 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %220 = stablehlo.convolution(%218, %219) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<8x16x16x160xf32>
    %221 = "mhlo.fusion"(%220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      mhlo.return %413 : tensor<8x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<8x256x160xf32>
    %222 = "mhlo.fusion"(%221, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256xf32>
    %223 = "mhlo.fusion"(%222, %220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x160xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x160xf32>
      mhlo.return %418 : tensor<8x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<8x256x160xf32>
    %224 = "mhlo.fusion"(%223, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256x5xf32>
      mhlo.return %407 : tensor<8x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256x5xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x5xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x5xf32>) -> tensor<8x256xf32>
    %226 = "mhlo.fusion"(%arg114, %arg115, %225, %222, %220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x160xf32>, %arg213: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x256x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x256x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x160xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x160xf32>) -> tensor<2048x160xf32>
      mhlo.return %424 : tensor<2048x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<2048x160xf32>
    %227 = stablehlo.dot_general %226, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x160xf32>, tensor<160x160xf32>) -> tensor<2048x160xf32>
    %228 = "mhlo.fusion"(%227, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<2048x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x160xf32>) -> tensor<8x256x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,32,256]{2,1,3,0}"} : (tensor<8x256x5x32xf32>) -> tensor<8x5x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x5x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x32x256xf32>) -> tensor<40x32x256xf32>
      mhlo.return %412 : tensor<40x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x160xf32>, tensor<160xf32>) -> tensor<40x32x256xf32>
    %229 = stablehlo.dot_general %217, %228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<40x1024x32xf32>, tensor<40x32x256xf32>) -> tensor<40x1024x256xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<40x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<40x1024x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<40x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
      mhlo.return %409 : tensor<8x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
    %231 = "mhlo.fusion"(%230, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024x8xf32>
      mhlo.return %407 : tensor<8x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024x8xf32>
    %232 = "mhlo.fusion"(%231, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x5x1024x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x5x1024x8xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
      mhlo.return %407 : tensor<8x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x8xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
    %233 = "mhlo.fusion"(%232, %229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024xf32>, %arg209: tensor<40x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<40x1024x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<40x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x5x1024xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x5x1024xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x5x1024xf32>) -> tensor<8x5x1024x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x5x1024x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x5x1024x256xf32>
      mhlo.return %414 : tensor<8x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024xf32>, tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
    %234 = "mhlo.fusion"(%233, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
      mhlo.return %407 : tensor<8x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
    %235 = "mhlo.fusion"(%233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<8x5x1024xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x5x1024xf32>) -> tensor<8x5x1024x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x5x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x1024x256xf32>) -> tensor<40x1024x256xf32>
      mhlo.return %409 : tensor<40x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<8x5x1024xf32>) -> tensor<40x1024x256xf32>
    %236 = stablehlo.dot_general %226, %arg118, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x160xf32>, tensor<160x160xf32>) -> tensor<2048x160xf32>
    %237 = "mhlo.fusion"(%236, %arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<2048x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x160xf32>) -> tensor<8x256x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,256,32]{3,1,2,0}"} : (tensor<8x256x5x32xf32>) -> tensor<8x5x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x5x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x256x32xf32>) -> tensor<40x256x32xf32>
      mhlo.return %412 : tensor<40x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x160xf32>, tensor<160xf32>) -> tensor<40x256x32xf32>
    %238 = stablehlo.dot_general %235, %237, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<40x1024x256xf32>, tensor<40x256x32xf32>) -> tensor<40x1024x32xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<40x1024x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x32xf32>) -> tensor<8x5x1024x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,5,32]{3,1,2,0}"} : (tensor<8x5x1024x32xf32>) -> tensor<8x1024x5x32xf32>
      %409 = mhlo.copy %408 : tensor<8x1024x5x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x5x32xf32>) -> tensor<8192x160xf32>
      mhlo.return %410 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<40x1024x32xf32>) -> tensor<8192x160xf32>
    %240 = stablehlo.dot_general %239, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<160x160xf32>) -> tensor<8192x160xf32>
    %241 = "mhlo.fusion"(%240, %arg106, %arg107, %208, %204, %202, %arg105, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8192x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<8x1024xf32>, %arg212: tensor<8x1024xf32>, %arg213: tensor<8x32x32x160xf32>, %arg214: tensor<160xf32>, %arg215: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<8x32x32x160xf32>
      %411 = stablehlo.add %arg213, %410 : tensor<8x32x32x160xf32>
      %412 = stablehlo.transpose %411, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %413 = mhlo.copy %412 : tensor<8x160x32x32xf32>
      %414 = mhlo.bitcast %413 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      %415 = stablehlo.transpose %414, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,160]{1,2,0}"} : (tensor<8x160x1024xf32>) -> tensor<8x1024x160xf32>
      %416 = mhlo.copy %415 : tensor<8x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<8x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %420 = stablehlo.subtract %416, %419 : tensor<8x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<8x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<8x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<8x1024x160xf32>
      %427 = stablehlo.add %409, %426 : tensor<8x1024x160xf32>
      mhlo.return %427 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8192x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, tensor<8x32x32x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %242 = "mhlo.fusion"(%241, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %243 = "mhlo.fusion"(%241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x1024x160xf32>
      mhlo.return %411 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
    %244 = "mhlo.fusion"(%243, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %245 = "mhlo.fusion"(%244, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %246 = "mhlo.fusion"(%245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %247 = "mhlo.fusion"(%arg122, %arg123, %246, %241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8192x160xf32>
      mhlo.return %417 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8192x160xf32>
    %248 = stablehlo.dot_general %247, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<640x160xf32>) -> tensor<8192x640xf32>
    %249 = "mhlo.fusion"(%248, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8192x640xf32>, %arg209: tensor<640xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<8192x640xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<8192x640xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x640xf32>) -> tensor<8x1024x640xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,640,1024]{1,2,0}"} : (tensor<8x1024x640xf32>) -> tensor<8x640x1024xf32>
      %411 = mhlo.copy %410 : tensor<8x640x1024xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x640x1024xf32>) -> tensor<8x640x32x32xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,32,640]{2,1,3,0}"} : (tensor<8x640x32x32xf32>) -> tensor<8x32x32x640xf32>
      %414 = mhlo.copy %413 : tensor<8x32x32x640xf32>
      mhlo.return %414 : tensor<8x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<8192x640xf32>, tensor<640xf32>) -> tensor<8x32x32x640xf32>
    %250 = "mhlo.fusion"(%arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x640xf32>
      mhlo.return %408 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %251 = stablehlo.convolution(%249, %250) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<8x32x32x640xf32>
    %252 = "mhlo.fusion"(%251, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x32x32x640xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<8x32x32x640xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x32x32x640xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x32x32x640xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,640,32,32]{1,3,2,0}"} : (tensor<8x32x32x640xf32>) -> tensor<8x640x32x32xf32>
      %412 = mhlo.copy %411 : tensor<8x640x32x32xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x640x32x32xf32>) -> tensor<8x640x1024xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,640]{1,2,0}"} : (tensor<8x640x1024xf32>) -> tensor<8x1024x640xf32>
      %415 = mhlo.copy %414 : tensor<8x1024x640xf32>
      %416 = stablehlo.negate %409 : tensor<8x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x32x32x640xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x32x32x640xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,640,32,32]{1,3,2,0}"} : (tensor<8x32x32x640xf32>) -> tensor<8x640x32x32xf32>
      %420 = mhlo.copy %419 : tensor<8x640x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x640x32x32xf32>) -> tensor<8x640x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,640]{1,2,0}"} : (tensor<8x640x1024xf32>) -> tensor<8x1024x640xf32>
      %423 = mhlo.copy %422 : tensor<8x1024x640xf32>
      %424 = stablehlo.abs %423 : tensor<8x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x1024x640xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x1024x640xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x1024x640xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x1024x640xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x1024x640xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x1024x640xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x1024x640xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x1024x640xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %452 = stablehlo.negate %428 : tensor<8x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %455 = stablehlo.exponential %452 : tensor<8x1024x640xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x1024x640xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x1024x640xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x1024x640xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x1024x640xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x1024x640xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x1024x640xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x1024x640xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x1024x640xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x1024x640xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x1024x640xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x1024x640xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x1024x640xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x1024x640xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x1024x640xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x1024x640xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x1024x640xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x1024x640xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x1024x640xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x1024x640xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x1024x640xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x640xf32>) -> tensor<8192x640xf32>
      mhlo.return %514 : tensor<8192x640xf32>
    }) {output_operand_aliasing = []} : (tensor<8x32x32x640xf32>, tensor<640xf32>) -> tensor<8192x640xf32>
    %253 = stablehlo.dot_general %252, %arg128, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x640xf32>, tensor<160x640xf32>) -> tensor<8192x160xf32>
    %254 = "mhlo.fusion"(%241, %253, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8192x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x1024x160xf32>
      mhlo.return %410 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8192x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %255 = "mhlo.fusion"(%254, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %256 = "mhlo.fusion"(%254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x1024x160xf32>
      mhlo.return %411 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
    %257 = "mhlo.fusion"(%256, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %258 = "mhlo.fusion"(%257, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %260 = "mhlo.fusion"(%arg130, %arg131, %259, %254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8192x160xf32>
      mhlo.return %417 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8192x160xf32>
    %261 = stablehlo.dot_general %260, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<160x160xf32>) -> tensor<8192x160xf32>
    %262 = "mhlo.fusion"(%261, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8192x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,1024,32]{3,1,2,0}"} : (tensor<8x1024x5x32xf32>) -> tensor<8x5x1024x32xf32>
      %411 = mhlo.copy %410 : tensor<8x5x1024x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x1024x32xf32>) -> tensor<40x1024x32xf32>
      mhlo.return %412 : tensor<40x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8192x160xf32>, tensor<160xf32>) -> tensor<40x1024x32xf32>
    %263 = "mhlo.fusion"(%arg130, %arg131, %259, %254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = stablehlo.transpose %416, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,160,1024]{1,2,0}"} : (tensor<8x1024x160xf32>) -> tensor<8x160x1024xf32>
      %418 = mhlo.copy %417 : tensor<8x160x1024xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x160x1024xf32>) -> tensor<8x160x32x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,32,160]{2,1,3,0}"} : (tensor<8x160x32x32xf32>) -> tensor<8x32x32x160xf32>
      %421 = mhlo.copy %420 : tensor<8x32x32x160xf32>
      mhlo.return %421 : tensor<8x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x32x32x160xf32>
    %264 = "mhlo.fusion"(%arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %408 = mhlo.copy %407 : tensor<2x2x160x160xf32>
      mhlo.return %408 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %265 = stablehlo.convolution(%263, %264) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<8x16x16x160xf32>
    %266 = "mhlo.fusion"(%265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      mhlo.return %413 : tensor<8x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<8x256x160xf32>
    %267 = "mhlo.fusion"(%266, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256xf32>
    %268 = "mhlo.fusion"(%267, %265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x160xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x160xf32>
      mhlo.return %418 : tensor<8x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<8x256x160xf32>
    %269 = "mhlo.fusion"(%268, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256x5xf32>
      mhlo.return %407 : tensor<8x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x160xf32>, tensor<f32>) -> tensor<8x256x5xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x5xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x5xf32>) -> tensor<8x256xf32>
    %271 = "mhlo.fusion"(%arg136, %arg137, %270, %267, %265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x160xf32>, %arg213: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<8x16x16x160xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x160xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,16,16]{1,3,2,0}"} : (tensor<8x16x16x160xf32>) -> tensor<8x160x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x160x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x16x16xf32>) -> tensor<8x160x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,160]{1,2,0}"} : (tensor<8x160x256xf32>) -> tensor<8x256x160xf32>
      %413 = mhlo.copy %412 : tensor<8x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x160xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x160xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x256x160xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x160xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x256x160xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x160xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x160xf32>) -> tensor<2048x160xf32>
      mhlo.return %424 : tensor<2048x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x160xf32>, tensor<160xf32>) -> tensor<2048x160xf32>
    %272 = stablehlo.dot_general %271, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x160xf32>, tensor<160x160xf32>) -> tensor<2048x160xf32>
    %273 = "mhlo.fusion"(%272, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<2048x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x160xf32>) -> tensor<8x256x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,32,256]{2,1,3,0}"} : (tensor<8x256x5x32xf32>) -> tensor<8x5x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x5x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x32x256xf32>) -> tensor<40x32x256xf32>
      mhlo.return %412 : tensor<40x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x160xf32>, tensor<160xf32>) -> tensor<40x32x256xf32>
    %274 = stablehlo.dot_general %262, %273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<40x1024x32xf32>, tensor<40x32x256xf32>) -> tensor<40x1024x256xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<40x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<40x1024x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<40x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
      mhlo.return %409 : tensor<8x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024x8xf32>
      mhlo.return %407 : tensor<8x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024x8xf32>
    %277 = "mhlo.fusion"(%276, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x5x1024x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x5x1024x8xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
      mhlo.return %407 : tensor<8x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x8xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
    %278 = "mhlo.fusion"(%277, %274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024xf32>, %arg209: tensor<40x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<40x1024x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<40x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x5x1024xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x5x1024xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x5x1024xf32>) -> tensor<8x5x1024x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x5x1024x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x5x1024x256xf32>
      mhlo.return %414 : tensor<8x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024xf32>, tensor<40x1024x256xf32>) -> tensor<8x5x1024x256xf32>
    %279 = "mhlo.fusion"(%278, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
      mhlo.return %407 : tensor<8x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<f32>) -> tensor<8x5x1024xf32>
    %280 = "mhlo.fusion"(%278, %279) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x5x1024x256xf32>, %arg209: tensor<8x5x1024xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x5x1024xf32>) -> tensor<8x5x1024x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x5x1024x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x1024x256xf32>) -> tensor<40x1024x256xf32>
      mhlo.return %409 : tensor<40x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x5x1024x256xf32>, tensor<8x5x1024xf32>) -> tensor<40x1024x256xf32>
    %281 = stablehlo.dot_general %271, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x160xf32>, tensor<160x160xf32>) -> tensor<2048x160xf32>
    %282 = "mhlo.fusion"(%281, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x160xf32>, %arg209: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<2048x160xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x160xf32>) -> tensor<8x256x5x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,5,256,32]{3,1,2,0}"} : (tensor<8x256x5x32xf32>) -> tensor<8x5x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x5x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x5x256x32xf32>) -> tensor<40x256x32xf32>
      mhlo.return %412 : tensor<40x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x160xf32>, tensor<160xf32>) -> tensor<40x256x32xf32>
    %283 = stablehlo.dot_general %280, %282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<40x1024x256xf32>, tensor<40x256x32xf32>) -> tensor<40x1024x32xf32>
    %284 = "mhlo.fusion"(%283) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<40x1024x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<40x1024x32xf32>) -> tensor<8x5x1024x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,5,32]{3,1,2,0}"} : (tensor<8x5x1024x32xf32>) -> tensor<8x1024x5x32xf32>
      %409 = mhlo.copy %408 : tensor<8x1024x5x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x5x32xf32>) -> tensor<8192x160xf32>
      mhlo.return %410 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<40x1024x32xf32>) -> tensor<8192x160xf32>
    %285 = stablehlo.dot_general %284, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<160x160xf32>) -> tensor<8192x160xf32>
    %286 = "mhlo.fusion"(%254, %285, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8192x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x1024x160xf32>
      mhlo.return %410 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8192x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %287 = "mhlo.fusion"(%286, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %288 = "mhlo.fusion"(%286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x1024x160xf32>
      mhlo.return %411 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
    %289 = "mhlo.fusion"(%288, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %290 = "mhlo.fusion"(%289, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %292 = "mhlo.fusion"(%arg144, %arg145, %291, %286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8192x160xf32>
      mhlo.return %417 : tensor<8192x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8192x160xf32>
    %293 = stablehlo.dot_general %292, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x160xf32>, tensor<640x160xf32>) -> tensor<8192x640xf32>
    %294 = "mhlo.fusion"(%293, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8192x640xf32>, %arg209: tensor<640xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<8192x640xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<8192x640xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x640xf32>) -> tensor<8x1024x640xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,640,1024]{1,2,0}"} : (tensor<8x1024x640xf32>) -> tensor<8x640x1024xf32>
      %411 = mhlo.copy %410 : tensor<8x640x1024xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x640x1024xf32>) -> tensor<8x640x32x32xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,32,32,640]{2,1,3,0}"} : (tensor<8x640x32x32xf32>) -> tensor<8x32x32x640xf32>
      %414 = mhlo.copy %413 : tensor<8x32x32x640xf32>
      mhlo.return %414 : tensor<8x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<8192x640xf32>, tensor<640xf32>) -> tensor<8x32x32x640xf32>
    %295 = "mhlo.fusion"(%arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x640xf32>
      mhlo.return %408 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %296 = stablehlo.convolution(%294, %295) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<8x32x32x640xf32>
    %297 = "mhlo.fusion"(%296, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x32x32x640xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<8x32x32x640xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x32x32x640xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x32x32x640xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,640,32,32]{1,3,2,0}"} : (tensor<8x32x32x640xf32>) -> tensor<8x640x32x32xf32>
      %412 = mhlo.copy %411 : tensor<8x640x32x32xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x640x32x32xf32>) -> tensor<8x640x1024xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,640]{1,2,0}"} : (tensor<8x640x1024xf32>) -> tensor<8x1024x640xf32>
      %415 = mhlo.copy %414 : tensor<8x1024x640xf32>
      %416 = stablehlo.negate %409 : tensor<8x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x32x32x640xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x32x32x640xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,640,32,32]{1,3,2,0}"} : (tensor<8x32x32x640xf32>) -> tensor<8x640x32x32xf32>
      %420 = mhlo.copy %419 : tensor<8x640x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x640x32x32xf32>) -> tensor<8x640x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,640]{1,2,0}"} : (tensor<8x640x1024xf32>) -> tensor<8x1024x640xf32>
      %423 = mhlo.copy %422 : tensor<8x1024x640xf32>
      %424 = stablehlo.abs %423 : tensor<8x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x1024x640xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x1024x640xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x1024x640xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x1024x640xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x1024x640xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x1024x640xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x1024x640xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x1024x640xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %452 = stablehlo.negate %428 : tensor<8x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %455 = stablehlo.exponential %452 : tensor<8x1024x640xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x1024x640xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x1024x640xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x1024x640xf32>, tensor<8x1024x640xf32>) -> tensor<8x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x1024x640xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x1024x640xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x1024x640xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x1024x640xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x1024x640xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x1024x640xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x1024x640xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x1024x640xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x1024x640xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x1024x640xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x1024x640xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x1024x640xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x1024x640xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x1024x640xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x1024x640xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x1024x640xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x1024x640xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x1024x640xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x1024x640xi1>, tensor<8x1024x640xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x1024x640xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x640xf32>) -> tensor<8192x640xf32>
      mhlo.return %514 : tensor<8192x640xf32>
    }) {output_operand_aliasing = []} : (tensor<8x32x32x640xf32>, tensor<640xf32>) -> tensor<8192x640xf32>
    %298 = stablehlo.dot_general %297, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8192x640xf32>, tensor<160x640xf32>) -> tensor<8192x160xf32>
    %299 = "mhlo.fusion"(%286, %298, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8192x160xf32>, %arg210: tensor<160xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<8192x160xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<8192x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8192x160xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x1024x160xf32>
      mhlo.return %410 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8192x160xf32>, tensor<160xf32>) -> tensor<8x1024x160xf32>
    %300 = "mhlo.fusion"(%299, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %301 = "mhlo.fusion"(%299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x1024x160xf32>
      mhlo.return %411 : tensor<8x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
    %302 = "mhlo.fusion"(%301, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024x160xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
      mhlo.return %407 : tensor<8x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x160xf32>, tensor<f32>) -> tensor<8x1024x5xf32>
    %303 = "mhlo.fusion"(%302, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x1024x5xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
      mhlo.return %407 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024x5xf32>, tensor<f32>) -> tensor<8x1024xf32>
    %304 = "mhlo.fusion"(%303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<8x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %410 = stablehlo.add %408, %409 : tensor<8x1024xf32>
      %411 = stablehlo.rsqrt %410 : tensor<8x1024xf32>
      mhlo.return %411 : tensor<8x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %305 = "mhlo.fusion"(%arg152, %arg153, %304, %299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8x32x32x160xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,160,32,32]{1,3,2,0}"} : (tensor<8x32x32x160xf32>) -> tensor<8x160x32x32xf32>
      %419 = mhlo.copy %418 : tensor<8x160x32x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x160x32x32xf32>) -> tensor<8x160x1024xf32>
      mhlo.return %420 : tensor<8x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x160x1024xf32>
    %306 = "mhlo.fusion"(%arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>):
      %407 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[160,256]{0,1}"} : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %408 = mhlo.copy %407 : tensor<160x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<8x160x256xf32>
      mhlo.return %409 : tensor<8x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>) -> tensor<8x160x256xf32>
    %307 = stablehlo.dot_general %305, %306, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x160x1024xf32>, tensor<8x160x256xf32>) -> tensor<8x1024x256xf32>
    %308 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<15> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %409 = stablehlo.iota dim = 0 : tensor<128xi32>
      %410 = stablehlo.convert %409 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %412 = stablehlo.add %410, %411 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %416 = stablehlo.add %414, %415 : tensor<128xf32>
      %417 = stablehlo.maximum %408, %416 : tensor<128xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %419 = stablehlo.convert %418 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %420 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %421 = stablehlo.add %419, %420 : tensor<128x1xi32>
      %422 = stablehlo.minimum %407, %421 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %423 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %424 = stablehlo.compare LT, %422, %423 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<16> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %422, %425 : tensor<128x1xi32>
      %427 = stablehlo.select %424, %426, %422 : tensor<128x1xi1>, tensor<128x1xi32>
      %428 = mhlo.bitcast %427 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %430 = stablehlo.convert %417 : (tensor<128xf32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %433 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.add %430, %433 : tensor<128xi32>
      %435 = stablehlo.select %432, %434, %430 : tensor<128xi1>, tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %437 = stablehlo.concatenate %429, %436, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %437 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %309 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %408 = stablehlo.iota dim = 0 : tensor<128xi32>
      %409 = stablehlo.convert %408 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %411 = stablehlo.add %409, %410 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.add %413, %414 : tensor<128xf32>
      %416 = stablehlo.maximum %407, %415 : tensor<128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %418 = stablehlo.convert %417 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %419 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %420 = stablehlo.compare LT, %418, %419 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<16> : tensor<i32>
      %421 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %422 = stablehlo.add %418, %421 : tensor<128x1xi32>
      %423 = stablehlo.select %420, %422, %418 : tensor<128x1xi1>, tensor<128x1xi32>
      %424 = mhlo.bitcast %423 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %426 = stablehlo.convert %416 : (tensor<128xf32>) -> tensor<128xi32>
      %427 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %428 = stablehlo.compare LT, %426, %427 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %429 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %430 = stablehlo.add %426, %429 : tensor<128xi32>
      %431 = stablehlo.select %428, %430, %426 : tensor<128xi1>, tensor<128xi32>
      %432 = stablehlo.broadcast_in_dim %431, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %433 = stablehlo.concatenate %425, %432, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %433 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %310 = "mhlo.fusion"(%arg152, %arg153, %304, %299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<8x1024xf32>, %arg211: tensor<8x1024x160xf32>, %arg212: tensor<8x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1024xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x1024xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x1024xf32>) -> tensor<8x1024x160xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x1024x160xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x1024x160xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<8x1024x160xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x1024x160xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x160xf32>) -> tensor<8x32x32x160xf32>
      mhlo.return %417 : tensor<8x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<8x1024xf32>, tensor<8x1024x160xf32>, tensor<8x1024xf32>) -> tensor<8x32x32x160xf32>
    %311 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,160,256]{1,0,2,3}"} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
      %408 = mhlo.copy %407 : tensor<3x3x160x256xf32>
      mhlo.return %408 : tensor<3x3x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
    %312 = stablehlo.convolution(%310, %311) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x160xf32>, tensor<3x3x160x256xf32>) -> tensor<8x16x16x256xf32>
    %313 = "mhlo.fusion"(%312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x16x16x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x256x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %413 = mhlo.copy %412 : tensor<8x256x256xf32>
      mhlo.return %413 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %314 = "mhlo.fusion"(%313, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %315 = "mhlo.fusion"(%314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<8x16x16x256xf32>, %arg210: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %408 = stablehlo.add %arg209, %407 : tensor<8x16x16x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x256x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %413 = mhlo.copy %412 : tensor<8x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x256xf32>
      %418 = stablehlo.multiply %417, %417 : tensor<8x256x256xf32>
      mhlo.return %418 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<8x16x16x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %316 = "mhlo.fusion"(%315, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %317 = "mhlo.fusion"(%316) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %318 = "mhlo.fusion"(%arg157, %arg158, %317, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x16x16x256xf32>, %arg213: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %408 = stablehlo.add %arg212, %407 : tensor<8x16x16x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x256x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %413 = mhlo.copy %412 : tensor<8x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x256xf32>
      %418 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x256xf32>
      %420 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x256xf32>
      %422 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x256xf32>
      mhlo.return %423 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %319 = "mhlo.fusion"(%318, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %320 = "mhlo.fusion"(%319, %arg157, %arg158, %317, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x256xf32>, %arg213: tensor<8x16x16x256xf32>, %arg214: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %408 = stablehlo.add %arg213, %407 : tensor<8x16x16x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x256x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %413 = mhlo.copy %412 : tensor<8x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg212, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x256xf32>
      %418 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x256xf32>
      %420 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x256xf32>
      %422 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x256xf32>
      %424 = stablehlo.multiply %arg208, %414 : tensor<8x256xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x256x256xf32>
      %427 = stablehlo.multiply %426, %426 : tensor<8x256x256xf32>
      mhlo.return %427 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %322 = "mhlo.fusion"(%321) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %323 = "mhlo.fusion"(%arg159, %arg160, %322, %319, %arg157, %arg158, %317, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<8x256xf32>, %arg215: tensor<8x256xf32>, %arg216: tensor<8x16x16x256xf32>, %arg217: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %408 = stablehlo.add %arg216, %407 : tensor<8x16x16x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %410 = mhlo.copy %409 : tensor<8x256x16x16xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %413 = mhlo.copy %412 : tensor<8x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %415 = stablehlo.multiply %arg215, %414 : tensor<8x256xf32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %417 = stablehlo.subtract %413, %416 : tensor<8x256x256xf32>
      %418 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<8x256x256xf32>
      %420 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %421 = stablehlo.multiply %419, %420 : tensor<8x256x256xf32>
      %422 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %423 = stablehlo.add %421, %422 : tensor<8x256x256xf32>
      %424 = stablehlo.multiply %arg211, %414 : tensor<8x256xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %426 = stablehlo.subtract %423, %425 : tensor<8x256x256xf32>
      %427 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<8x256x256xf32>
      %429 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<8x256x256xf32>
      %431 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %432 = stablehlo.add %430, %431 : tensor<8x256x256xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<2048x256xf32>
      mhlo.return %433 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x256xf32>, tensor<256xf32>) -> tensor<2048x256xf32>
    %324 = stablehlo.dot_general %323, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %325 = "mhlo.fusion"(%324, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,256,32]{3,1,2,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x8x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %412 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x256x32xf32>
    %326 = stablehlo.dot_general %323, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %327 = "mhlo.fusion"(%326, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,32,256]{2,1,3,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x8x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x32x256xf32>) -> tensor<64x32x256xf32>
      mhlo.return %412 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x32x256xf32>
    %328 = stablehlo.dot_general %325, %327, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x32xf32>, tensor<64x32x256xf32>) -> tensor<64x256x256xf32>
    %329 = "mhlo.fusion"(%328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<64x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
      mhlo.return %409 : tensor<8x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
    %330 = "mhlo.fusion"(%329, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256x8xf32>
      mhlo.return %407 : tensor<8x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256x8xf32>
    %331 = "mhlo.fusion"(%330, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x8x256x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x8x256x8xf32>, tensor<f32>) -> tensor<8x8x256xf32>
      mhlo.return %407 : tensor<8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x8xf32>, tensor<f32>) -> tensor<8x8x256xf32>
    %332 = "mhlo.fusion"(%331, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256xf32>, %arg209: tensor<64x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<64x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x8x256xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x8x256xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x8x256xf32>) -> tensor<8x8x256x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x8x256x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x8x256x256xf32>
      mhlo.return %414 : tensor<8x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256xf32>, tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
    %333 = "mhlo.fusion"(%332, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256xf32>
      mhlo.return %407 : tensor<8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256xf32>
    %334 = "mhlo.fusion"(%332, %333) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<8x8x256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x8x256xf32>) -> tensor<8x8x256x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x8x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x256xf32>) -> tensor<64x256x256xf32>
      mhlo.return %409 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<8x8x256xf32>) -> tensor<64x256x256xf32>
    %335 = stablehlo.dot_general %323, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %336 = "mhlo.fusion"(%335, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,256,32]{3,1,2,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x8x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %412 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x256x32xf32>
    %337 = stablehlo.dot_general %334, %336, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x256xf32>, tensor<64x256x32xf32>) -> tensor<64x256x32xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<8x8x256x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,8,32]{3,1,2,0}"} : (tensor<8x8x256x32xf32>) -> tensor<8x256x8x32xf32>
      %409 = mhlo.copy %408 : tensor<8x256x8x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x8x32xf32>) -> tensor<2048x256xf32>
      mhlo.return %410 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x32xf32>) -> tensor<2048x256xf32>
    %339 = stablehlo.dot_general %338, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %340 = "mhlo.fusion"(%339, %arg157, %arg158, %317, %314, %312, %arg156, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<8x256xf32>, %arg212: tensor<8x256xf32>, %arg213: tensor<8x16x16x256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<8x16x16x256xf32>
      %411 = stablehlo.add %arg213, %410 : tensor<8x16x16x256xf32>
      %412 = stablehlo.transpose %411, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %413 = mhlo.copy %412 : tensor<8x256x16x16xf32>
      %414 = mhlo.bitcast %413 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %416 = mhlo.copy %415 : tensor<8x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<8x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %420 = stablehlo.subtract %416, %419 : tensor<8x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<8x256x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<8x256x256xf32>
      %425 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %426 = stablehlo.add %424, %425 : tensor<8x256x256xf32>
      %427 = stablehlo.add %409, %426 : tensor<8x256x256xf32>
      mhlo.return %427 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256xf32>, tensor<8x16x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %341 = "mhlo.fusion"(%340, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %342 = "mhlo.fusion"(%340, %341) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x256x256xf32>
      mhlo.return %411 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %343 = "mhlo.fusion"(%342, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %344 = "mhlo.fusion"(%343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %345 = "mhlo.fusion"(%arg169, %arg170, %344, %340, %341) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256x256xf32>, %arg212: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x256x256xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x256x256xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x256x256xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<2048x256xf32>
      mhlo.return %417 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<2048x256xf32>
    %346 = stablehlo.dot_general %345, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<1024x256xf32>) -> tensor<2048x1024xf32>
    %347 = "mhlo.fusion"(%346, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x1024xf32>, %arg209: tensor<1024xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<2048x1024xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x1024xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x1024xf32>) -> tensor<8x256x1024xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,256]{1,2,0}"} : (tensor<8x256x1024xf32>) -> tensor<8x1024x256xf32>
      %411 = mhlo.copy %410 : tensor<8x1024x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x256xf32>) -> tensor<8x1024x16x16xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,16,16,1024]{2,1,3,0}"} : (tensor<8x1024x16x16xf32>) -> tensor<8x16x16x1024xf32>
      %414 = mhlo.copy %413 : tensor<8x16x16x1024xf32>
      mhlo.return %414 : tensor<8x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8x16x16x1024xf32>
    %348 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x1024xf32>
      mhlo.return %408 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %349 = stablehlo.convolution(%347, %348) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<8x16x16x1024xf32>
    %350 = "mhlo.fusion"(%349, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16x16x1024xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<8x16x16x1024xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x16x16x1024xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x16x16x1024xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,16,16]{1,3,2,0}"} : (tensor<8x16x16x1024xf32>) -> tensor<8x1024x16x16xf32>
      %412 = mhlo.copy %411 : tensor<8x1024x16x16xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x16x16xf32>) -> tensor<8x1024x256xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,1024]{1,2,0}"} : (tensor<8x1024x256xf32>) -> tensor<8x256x1024xf32>
      %415 = mhlo.copy %414 : tensor<8x256x1024xf32>
      %416 = stablehlo.negate %409 : tensor<8x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x16x16x1024xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x16x16x1024xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,16,16]{1,3,2,0}"} : (tensor<8x16x16x1024xf32>) -> tensor<8x1024x16x16xf32>
      %420 = mhlo.copy %419 : tensor<8x1024x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x16x16xf32>) -> tensor<8x1024x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,1024]{1,2,0}"} : (tensor<8x1024x256xf32>) -> tensor<8x256x1024xf32>
      %423 = mhlo.copy %422 : tensor<8x256x1024xf32>
      %424 = stablehlo.abs %423 : tensor<8x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x256x1024xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x256x1024xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x256x1024xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x256x1024xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x256x1024xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x256x1024xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x256x1024xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x256x1024xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %452 = stablehlo.negate %428 : tensor<8x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %455 = stablehlo.exponential %452 : tensor<8x256x1024xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x256x1024xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x256x1024xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x256x1024xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x256x1024xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x256x1024xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x256x1024xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x256x1024xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x256x1024xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x256x1024xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x256x1024xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x256x1024xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x256x1024xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x256x1024xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x256x1024xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x256x1024xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x256x1024xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x256x1024xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x256x1024xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x256x1024xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x256x1024xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x1024xf32>) -> tensor<2048x1024xf32>
      mhlo.return %514 : tensor<2048x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x1024xf32>, tensor<1024xf32>) -> tensor<2048x1024xf32>
    %351 = stablehlo.dot_general %350, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x1024xf32>, tensor<256x1024xf32>) -> tensor<2048x256xf32>
    %352 = "mhlo.fusion"(%340, %351, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<2048x256xf32>, %arg210: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x256x256xf32>
      mhlo.return %410 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<2048x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %353 = "mhlo.fusion"(%352, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %354 = "mhlo.fusion"(%352, %353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x256x256xf32>
      mhlo.return %411 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %355 = "mhlo.fusion"(%354, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %356 = "mhlo.fusion"(%355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %357 = "mhlo.fusion"(%arg177, %arg178, %356, %352, %353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256x256xf32>, %arg212: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x256x256xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x256x256xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x256x256xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<2048x256xf32>
      mhlo.return %417 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<2048x256xf32>
    %358 = stablehlo.dot_general %357, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %359 = "mhlo.fusion"(%358, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,256,32]{3,1,2,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x8x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %412 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x256x32xf32>
    %360 = stablehlo.dot_general %357, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %361 = "mhlo.fusion"(%360, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,32,256]{2,1,3,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x32x256xf32>
      %411 = mhlo.copy %410 : tensor<8x8x32x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x32x256xf32>) -> tensor<64x32x256xf32>
      mhlo.return %412 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x32x256xf32>
    %362 = stablehlo.dot_general %359, %361, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x32xf32>, tensor<64x32x256xf32>) -> tensor<64x256x256xf32>
    %363 = "mhlo.fusion"(%362) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x256xf32>
      %408 = stablehlo.multiply %arg208, %407 : tensor<64x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
      mhlo.return %409 : tensor<8x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
    %364 = "mhlo.fusion"(%363, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256x8xf32>
      mhlo.return %407 : tensor<8x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256x8xf32>
    %365 = "mhlo.fusion"(%364, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x8x256x8xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<8x8x256x8xf32>, tensor<f32>) -> tensor<8x8x256xf32>
      mhlo.return %407 : tensor<8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x8xf32>, tensor<f32>) -> tensor<8x8x256xf32>
    %366 = "mhlo.fusion"(%365, %362) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256xf32>, %arg209: tensor<64x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<64x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x8x256xf32>
      %411 = stablehlo.maximum %410, %arg208 : tensor<8x8x256xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0, 1, 2] : (tensor<8x8x256xf32>) -> tensor<8x8x256x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<8x8x256x256xf32>
      %414 = stablehlo.exponential %413 : tensor<8x8x256x256xf32>
      mhlo.return %414 : tensor<8x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256xf32>, tensor<64x256x256xf32>) -> tensor<8x8x256x256xf32>
    %367 = "mhlo.fusion"(%366, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256xf32>
      mhlo.return %407 : tensor<8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<f32>) -> tensor<8x8x256xf32>
    %368 = "mhlo.fusion"(%366, %367) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x8x256x256xf32>, %arg209: tensor<8x8x256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<8x8x256xf32>) -> tensor<8x8x256x256xf32>
      %408 = stablehlo.divide %arg208, %407 : tensor<8x8x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x256xf32>) -> tensor<64x256x256xf32>
      mhlo.return %409 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x8x256x256xf32>, tensor<8x8x256xf32>) -> tensor<64x256x256xf32>
    %369 = stablehlo.dot_general %357, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %370 = "mhlo.fusion"(%369, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x256xf32>, %arg209: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x8x32xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,8,256,32]{3,1,2,0}"} : (tensor<8x256x8x32xf32>) -> tensor<8x8x256x32xf32>
      %411 = mhlo.copy %410 : tensor<8x8x256x32xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x8x256x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %412 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x256xf32>, tensor<256xf32>) -> tensor<64x256x32xf32>
    %371 = stablehlo.dot_general %368, %370, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x256xf32>, tensor<64x256x32xf32>) -> tensor<64x256x32xf32>
    %372 = "mhlo.fusion"(%371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x32xf32>):
      %407 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<8x8x256x32xf32>
      %408 = stablehlo.transpose %407, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,8,32]{3,1,2,0}"} : (tensor<8x8x256x32xf32>) -> tensor<8x256x8x32xf32>
      %409 = mhlo.copy %408 : tensor<8x256x8x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x8x32xf32>) -> tensor<2048x256xf32>
      mhlo.return %410 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x32xf32>) -> tensor<2048x256xf32>
    %373 = stablehlo.dot_general %372, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<256x256xf32>) -> tensor<2048x256xf32>
    %374 = "mhlo.fusion"(%352, %373, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<2048x256xf32>, %arg210: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x256x256xf32>
      mhlo.return %410 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<2048x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %375 = "mhlo.fusion"(%374, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %376 = "mhlo.fusion"(%374, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x256x256xf32>
      mhlo.return %411 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %377 = "mhlo.fusion"(%376, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %378 = "mhlo.fusion"(%377) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %379 = "mhlo.fusion"(%arg187, %arg188, %378, %374, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256x256xf32>, %arg212: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x256x256xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x256x256xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x256x256xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<2048x256xf32>
      mhlo.return %417 : tensor<2048x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<2048x256xf32>
    %380 = stablehlo.dot_general %379, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<1024x256xf32>) -> tensor<2048x1024xf32>
    %381 = "mhlo.fusion"(%380, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2048x1024xf32>, %arg209: tensor<1024xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<2048x1024xf32>
      %408 = stablehlo.add %407, %arg208 : tensor<2048x1024xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x1024xf32>) -> tensor<8x256x1024xf32>
      %410 = stablehlo.transpose %409, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,1024,256]{1,2,0}"} : (tensor<8x256x1024xf32>) -> tensor<8x1024x256xf32>
      %411 = mhlo.copy %410 : tensor<8x1024x256xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x1024x256xf32>) -> tensor<8x1024x16x16xf32>
      %413 = stablehlo.transpose %412, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,16,16,1024]{2,1,3,0}"} : (tensor<8x1024x16x16xf32>) -> tensor<8x16x16x1024xf32>
      %414 = mhlo.copy %413 : tensor<8x16x16x1024xf32>
      mhlo.return %414 : tensor<8x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<8x16x16x1024xf32>
    %382 = "mhlo.fusion"(%arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %408 = mhlo.copy %407 : tensor<3x3x1x1024xf32>
      mhlo.return %408 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %383 = stablehlo.convolution(%381, %382) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<8x16x16x1024xf32>
    %384 = "mhlo.fusion"(%383, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x16x16x1024xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<8x16x16x1024xf32>
      %409 = stablehlo.add %arg208, %408 : tensor<8x16x16x1024xf32>
      %410 = stablehlo.multiply %407, %409 : tensor<8x16x16x1024xf32>
      %411 = stablehlo.transpose %410, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,16,16]{1,3,2,0}"} : (tensor<8x16x16x1024xf32>) -> tensor<8x1024x16x16xf32>
      %412 = mhlo.copy %411 : tensor<8x1024x16x16xf32>
      %413 = mhlo.bitcast %412 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x16x16xf32>) -> tensor<8x1024x256xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,1024]{1,2,0}"} : (tensor<8x1024x256xf32>) -> tensor<8x256x1024xf32>
      %415 = mhlo.copy %414 : tensor<8x256x1024xf32>
      %416 = stablehlo.negate %409 : tensor<8x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x16x16x1024xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<8x16x16x1024xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,1024,16,16]{1,3,2,0}"} : (tensor<8x16x16x1024xf32>) -> tensor<8x1024x16x16xf32>
      %420 = mhlo.copy %419 : tensor<8x1024x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1024x16x16xf32>) -> tensor<8x1024x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,1024]{1,2,0}"} : (tensor<8x1024x256xf32>) -> tensor<8x256x1024xf32>
      %423 = mhlo.copy %422 : tensor<8x256x1024xf32>
      %424 = stablehlo.abs %423 : tensor<8x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %428 = stablehlo.multiply %423, %423 : tensor<8x256x1024xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<8x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %431 = stablehlo.add %429, %430 : tensor<8x256x1024xf32>
      %432 = stablehlo.multiply %431, %428 : tensor<8x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %434 = stablehlo.add %432, %433 : tensor<8x256x1024xf32>
      %435 = stablehlo.multiply %434, %428 : tensor<8x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<8x256x1024xf32>
      %438 = stablehlo.multiply %437, %428 : tensor<8x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<8x256x1024xf32>
      %441 = stablehlo.multiply %440, %428 : tensor<8x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<8x256x1024xf32>
      %444 = stablehlo.multiply %443, %428 : tensor<8x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<8x256x1024xf32>
      %447 = stablehlo.multiply %423, %446 : tensor<8x256x1024xf32>
      %448 = stablehlo.subtract %425, %447 : tensor<8x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %450 = stablehlo.compare LT, %423, %449 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %452 = stablehlo.negate %428 : tensor<8x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %454 = stablehlo.compare LT, %452, %453 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %455 = stablehlo.exponential %452 : tensor<8x256x1024xf32>
      %456 = stablehlo.divide %425, %424 : tensor<8x256x1024xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<8x256x1024xf32>
      %458 = stablehlo.compare LT, %424, %451 : (tensor<8x256x1024xf32>, tensor<8x256x1024xf32>) -> tensor<8x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %460 = stablehlo.divide %425, %428 : tensor<8x256x1024xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<8x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %463 = stablehlo.add %461, %462 : tensor<8x256x1024xf32>
      %464 = stablehlo.multiply %463, %460 : tensor<8x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %466 = stablehlo.add %464, %465 : tensor<8x256x1024xf32>
      %467 = stablehlo.multiply %466, %460 : tensor<8x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %469 = stablehlo.add %467, %468 : tensor<8x256x1024xf32>
      %470 = stablehlo.multiply %469, %460 : tensor<8x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<8x256x1024xf32>
      %473 = stablehlo.multiply %472, %460 : tensor<8x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<8x256x1024xf32>
      %476 = stablehlo.multiply %475, %460 : tensor<8x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<8x256x1024xf32>
      %479 = stablehlo.multiply %478, %460 : tensor<8x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<8x256x1024xf32>
      %482 = stablehlo.multiply %481, %460 : tensor<8x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<8x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %486 = stablehlo.multiply %485, %460 : tensor<8x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %488 = stablehlo.add %486, %487 : tensor<8x256x1024xf32>
      %489 = stablehlo.multiply %488, %460 : tensor<8x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %491 = stablehlo.add %489, %490 : tensor<8x256x1024xf32>
      %492 = stablehlo.multiply %491, %460 : tensor<8x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<8x256x1024xf32>
      %495 = stablehlo.multiply %494, %460 : tensor<8x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %497 = stablehlo.add %495, %496 : tensor<8x256x1024xf32>
      %498 = stablehlo.multiply %497, %460 : tensor<8x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %500 = stablehlo.add %498, %499 : tensor<8x256x1024xf32>
      %501 = stablehlo.multiply %500, %460 : tensor<8x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %503 = stablehlo.add %501, %502 : tensor<8x256x1024xf32>
      %504 = stablehlo.multiply %503, %460 : tensor<8x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x256x1024xf32>
      %506 = stablehlo.add %504, %505 : tensor<8x256x1024xf32>
      %507 = stablehlo.select %458, %484, %506 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %508 = stablehlo.multiply %457, %507 : tensor<8x256x1024xf32>
      %509 = stablehlo.select %454, %449, %508 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %510 = stablehlo.subtract %451, %509 : tensor<8x256x1024xf32>
      %511 = stablehlo.select %450, %510, %509 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %512 = stablehlo.select %426, %448, %511 : tensor<8x256x1024xi1>, tensor<8x256x1024xf32>
      %513 = stablehlo.multiply %415, %512 : tensor<8x256x1024xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x1024xf32>) -> tensor<2048x1024xf32>
      mhlo.return %514 : tensor<2048x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x16x1024xf32>, tensor<1024xf32>) -> tensor<2048x1024xf32>
    %385 = stablehlo.dot_general %384, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2048x1024xf32>, tensor<256x1024xf32>) -> tensor<2048x256xf32>
    %386 = "mhlo.fusion"(%374, %385, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<2048x256xf32>, %arg210: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<2048x256xf32>
      %408 = stablehlo.add %407, %arg209 : tensor<2048x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2048x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.add %409, %arg208 : tensor<8x256x256xf32>
      mhlo.return %410 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<2048x256xf32>, tensor<256xf32>) -> tensor<8x256x256xf32>
    %387 = "mhlo.fusion"(%386, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %407 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256xf32>
    %388 = "mhlo.fusion"(%386, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg209, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg208, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.multiply %410, %410 : tensor<8x256x256xf32>
      mhlo.return %411 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %389 = "mhlo.fusion"(%388, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>, %arg209: tensor<f32>):
      %407 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %408 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %408 : tensor<f32>
      }) : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
      mhlo.return %407 : tensor<8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>, tensor<f32>) -> tensor<8x256x8xf32>
    %390 = "mhlo.fusion"(%389) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<8x256x8xf32>, tensor<f32>) -> tensor<8x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %409 = stablehlo.multiply %407, %408 : tensor<8x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %411 = stablehlo.add %409, %410 : tensor<8x256xf32>
      %412 = stablehlo.rsqrt %411 : tensor<8x256xf32>
      mhlo.return %412 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8xf32>) -> tensor<8x256xf32>
    %391 = "mhlo.fusion"(%arg195, %arg196, %390, %386, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<8x256xf32>, %arg211: tensor<8x256x256xf32>, %arg212: tensor<8x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
      %408 = stablehlo.multiply %arg212, %407 : tensor<8x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %410 = stablehlo.subtract %arg211, %409 : tensor<8x256x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<8x256xf32>) -> tensor<8x256x256xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x256x256xf32>
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %414 = stablehlo.multiply %412, %413 : tensor<8x256x256xf32>
      %415 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<8x256x256xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<8x16x16x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,256,16,16]{1,3,2,0}"} : (tensor<8x16x16x256xf32>) -> tensor<8x256x16x16xf32>
      %419 = mhlo.copy %418 : tensor<8x256x16x16xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x16x16xf32>) -> tensor<8x256x256xf32>
      mhlo.return %420 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256xf32>, tensor<8x256x256xf32>, tensor<8x256xf32>) -> tensor<8x256x256xf32>
    %392 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %407 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %408 = mhlo.copy %407 : tensor<256x256xf32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<8x256x256xf32>
      mhlo.return %409 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<8x256x256xf32>
    %393 = stablehlo.dot_general %391, %392, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
    %394 = "mhlo.fusion"(%1, %2, %86, %arg198, %87, %88, %89, %197, %arg199, %198, %199, %307, %arg200, %cst_0, %308, %309, %c, %cst_1, %393, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x2xi32>, %arg209: tensor<128x128x2xi32>, %arg210: tensor<8x16384x256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128x128x2xi32>, %arg213: tensor<128x128x2xi32>, %arg214: tensor<128xf32>, %arg215: tensor<8x4096x256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<128x128x2xi32>, %arg218: tensor<128x128x2xi32>, %arg219: tensor<8x1024x256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<f32>, %arg222: tensor<128x128x2xi32>, %arg223: tensor<128x128x2xi32>, %arg224: tensor<i32>, %arg225: tensor<f32>, %arg226: tensor<8x256x256xf32>, %arg227: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg227, dims = [2] : (tensor<256xf32>) -> tensor<8x256x256xf32>
      %408 = stablehlo.add %arg226, %407 : tensor<8x256x256xf32>
      %409 = stablehlo.transpose %408, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,256]{1,2,0}"} : (tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
      %410 = mhlo.copy %409 : tensor<8x256x256xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<8x256x16x16xf32>
      %412 = mhlo.bitcast %arg223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %413 = "stablehlo.gather"(%411, %412) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %414 = mhlo.bitcast %413 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %415 = stablehlo.transpose %414, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %416 = mhlo.copy %415 : tensor<8x256x128x128xf32>
      %417 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.multiply %arg214, %418 : tensor<128xf32>
      %420 = stablehlo.broadcast_in_dim %arg225, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %422 = stablehlo.maximum %417, %421 : tensor<128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %424 = stablehlo.convert %423 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %425 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %427 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %428 = stablehlo.add %424, %427 : tensor<128x1xi32>
      %429 = stablehlo.select %426, %428, %424 : tensor<128x1xi1>, tensor<128x1xi32>
      %430 = mhlo.bitcast %429 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %432 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %433 = stablehlo.convert %422 : (tensor<128xf32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %434 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %435 = stablehlo.add %433, %434 : tensor<128xi32>
      %436 = stablehlo.minimum %432, %435 : tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %439 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %440 = stablehlo.add %436, %439 : tensor<128xi32>
      %441 = stablehlo.select %438, %440, %436 : tensor<128xi1>, tensor<128xi32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %443 = stablehlo.concatenate %431, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %444 = mhlo.bitcast %443 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %445 = "stablehlo.gather"(%411, %444) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %447 = stablehlo.transpose %446, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %448 = mhlo.copy %447 : tensor<8x256x128x128xf32>
      %449 = stablehlo.subtract %448, %416 : tensor<8x256x128x128xf32>
      %450 = stablehlo.convert %433 : (tensor<128xi32>) -> tensor<128xf32>
      %451 = stablehlo.subtract %422, %450 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %453 = stablehlo.clamp %417, %451, %452 : tensor<128xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [3] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %455 = stablehlo.multiply %449, %454 : tensor<8x256x128x128xf32>
      %456 = stablehlo.add %416, %455 : tensor<8x256x128x128xf32>
      %457 = mhlo.bitcast %arg222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %458 = "stablehlo.gather"(%411, %457) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %459 = mhlo.bitcast %458 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %460 = stablehlo.transpose %459, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %461 = mhlo.copy %460 : tensor<8x256x128x128xf32>
      %462 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %463 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %464 = stablehlo.add %424, %463 : tensor<128x1xi32>
      %465 = stablehlo.minimum %462, %464 : tensor<128x1xi32>
      %466 = stablehlo.compare LT, %465, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %467 = stablehlo.add %465, %427 : tensor<128x1xi32>
      %468 = stablehlo.select %466, %467, %465 : tensor<128x1xi1>, tensor<128x1xi32>
      %469 = mhlo.bitcast %468 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %470 = stablehlo.broadcast_in_dim %469, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %471 = stablehlo.concatenate %470, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %473 = "stablehlo.gather"(%411, %472) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %474 = mhlo.bitcast %473 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %475 = stablehlo.transpose %474, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %476 = mhlo.copy %475 : tensor<8x256x128x128xf32>
      %477 = stablehlo.subtract %476, %461 : tensor<8x256x128x128xf32>
      %478 = stablehlo.multiply %477, %454 : tensor<8x256x128x128xf32>
      %479 = stablehlo.add %461, %478 : tensor<8x256x128x128xf32>
      %480 = stablehlo.subtract %479, %456 : tensor<8x256x128x128xf32>
      %481 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %482 = stablehlo.convert %424 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %483 = stablehlo.subtract %423, %482 : tensor<128x1xf32>
      %484 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %485 = stablehlo.clamp %481, %483, %484 : tensor<128x1xf32>
      %486 = mhlo.bitcast %485 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %487 = stablehlo.broadcast_in_dim %486, dims = [2] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %488 = stablehlo.multiply %480, %487 : tensor<8x256x128x128xf32>
      %489 = stablehlo.add %456, %488 : tensor<8x256x128x128xf32>
      %490 = stablehlo.broadcast_in_dim %arg220, dims = [2] : (tensor<256xf32>) -> tensor<8x1024x256xf32>
      %491 = stablehlo.add %arg219, %490 : tensor<8x1024x256xf32>
      %492 = stablehlo.transpose %491, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,1024]{1,2,0}"} : (tensor<8x1024x256xf32>) -> tensor<8x256x1024xf32>
      %493 = mhlo.copy %492 : tensor<8x256x1024xf32>
      %494 = mhlo.bitcast %493 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x1024xf32>) -> tensor<8x256x32x32xf32>
      %495 = mhlo.bitcast %arg218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %496 = "stablehlo.gather"(%494, %495) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %497 = mhlo.bitcast %496 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %498 = stablehlo.transpose %497, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %499 = mhlo.copy %498 : tensor<8x256x128x128xf32>
      %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %501 = stablehlo.multiply %arg214, %500 : tensor<128xf32>
      %502 = stablehlo.add %501, %420 : tensor<128xf32>
      %503 = stablehlo.maximum %417, %502 : tensor<128xf32>
      %504 = mhlo.bitcast %503 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %505 = stablehlo.convert %504 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %506 = stablehlo.compare LT, %505, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<32> : tensor<i32>
      %507 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %508 = stablehlo.add %505, %507 : tensor<128x1xi32>
      %509 = stablehlo.select %506, %508, %505 : tensor<128x1xi1>, tensor<128x1xi32>
      %510 = mhlo.bitcast %509 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %511 = stablehlo.broadcast_in_dim %510, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_9 = stablehlo.constant dense<31> : tensor<i32>
      %512 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %513 = stablehlo.convert %503 : (tensor<128xf32>) -> tensor<128xi32>
      %514 = stablehlo.add %513, %434 : tensor<128xi32>
      %515 = stablehlo.minimum %512, %514 : tensor<128xi32>
      %516 = stablehlo.compare LT, %515, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %517 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %518 = stablehlo.add %515, %517 : tensor<128xi32>
      %519 = stablehlo.select %516, %518, %515 : tensor<128xi1>, tensor<128xi32>
      %520 = stablehlo.broadcast_in_dim %519, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %521 = stablehlo.concatenate %511, %520, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %522 = mhlo.bitcast %521 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %523 = "stablehlo.gather"(%494, %522) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %525 = stablehlo.transpose %524, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %526 = mhlo.copy %525 : tensor<8x256x128x128xf32>
      %527 = stablehlo.subtract %526, %499 : tensor<8x256x128x128xf32>
      %528 = stablehlo.convert %513 : (tensor<128xi32>) -> tensor<128xf32>
      %529 = stablehlo.subtract %503, %528 : tensor<128xf32>
      %530 = stablehlo.clamp %417, %529, %452 : tensor<128xf32>
      %531 = stablehlo.broadcast_in_dim %530, dims = [3] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %532 = stablehlo.multiply %527, %531 : tensor<8x256x128x128xf32>
      %533 = stablehlo.add %499, %532 : tensor<8x256x128x128xf32>
      %534 = mhlo.bitcast %arg217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %535 = "stablehlo.gather"(%494, %534) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %536 = mhlo.bitcast %535 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %537 = stablehlo.transpose %536, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %538 = mhlo.copy %537 : tensor<8x256x128x128xf32>
      %539 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %540 = stablehlo.add %505, %463 : tensor<128x1xi32>
      %541 = stablehlo.minimum %539, %540 : tensor<128x1xi32>
      %542 = stablehlo.compare LT, %541, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %543 = stablehlo.add %541, %507 : tensor<128x1xi32>
      %544 = stablehlo.select %542, %543, %541 : tensor<128x1xi1>, tensor<128x1xi32>
      %545 = mhlo.bitcast %544 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %546 = stablehlo.broadcast_in_dim %545, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %547 = stablehlo.concatenate %546, %520, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %548 = mhlo.bitcast %547 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %549 = "stablehlo.gather"(%494, %548) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %550 = mhlo.bitcast %549 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %551 = stablehlo.transpose %550, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %552 = mhlo.copy %551 : tensor<8x256x128x128xf32>
      %553 = stablehlo.subtract %552, %538 : tensor<8x256x128x128xf32>
      %554 = stablehlo.multiply %553, %531 : tensor<8x256x128x128xf32>
      %555 = stablehlo.add %538, %554 : tensor<8x256x128x128xf32>
      %556 = stablehlo.subtract %555, %533 : tensor<8x256x128x128xf32>
      %557 = stablehlo.convert %505 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %558 = stablehlo.subtract %504, %557 : tensor<128x1xf32>
      %559 = stablehlo.clamp %481, %558, %484 : tensor<128x1xf32>
      %560 = mhlo.bitcast %559 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %561 = stablehlo.broadcast_in_dim %560, dims = [2] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %562 = stablehlo.multiply %556, %561 : tensor<8x256x128x128xf32>
      %563 = stablehlo.add %533, %562 : tensor<8x256x128x128xf32>
      %564 = stablehlo.broadcast_in_dim %arg216, dims = [2] : (tensor<256xf32>) -> tensor<8x4096x256xf32>
      %565 = stablehlo.add %arg215, %564 : tensor<8x4096x256xf32>
      %566 = stablehlo.transpose %565, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,4096]{1,2,0}"} : (tensor<8x4096x256xf32>) -> tensor<8x256x4096xf32>
      %567 = mhlo.copy %566 : tensor<8x256x4096xf32>
      %568 = mhlo.bitcast %567 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x4096xf32>) -> tensor<8x256x64x64xf32>
      %569 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %570 = "stablehlo.gather"(%568, %569) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %571 = mhlo.bitcast %570 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %572 = stablehlo.transpose %571, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %573 = mhlo.copy %572 : tensor<8x256x128x128xf32>
      %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %574 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %575 = stablehlo.multiply %arg214, %574 : tensor<128xf32>
      %576 = stablehlo.add %575, %420 : tensor<128xf32>
      %577 = stablehlo.maximum %417, %576 : tensor<128xf32>
      %578 = mhlo.bitcast %577 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %579 = stablehlo.convert %578 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %580 = stablehlo.compare LT, %579, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<64> : tensor<i32>
      %581 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %582 = stablehlo.add %579, %581 : tensor<128x1xi32>
      %583 = stablehlo.select %580, %582, %579 : tensor<128x1xi1>, tensor<128x1xi32>
      %584 = mhlo.bitcast %583 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %585 = stablehlo.broadcast_in_dim %584, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_12 = stablehlo.constant dense<63> : tensor<i32>
      %586 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %587 = stablehlo.convert %577 : (tensor<128xf32>) -> tensor<128xi32>
      %588 = stablehlo.add %587, %434 : tensor<128xi32>
      %589 = stablehlo.minimum %586, %588 : tensor<128xi32>
      %590 = stablehlo.compare LT, %589, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %591 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %592 = stablehlo.add %589, %591 : tensor<128xi32>
      %593 = stablehlo.select %590, %592, %589 : tensor<128xi1>, tensor<128xi32>
      %594 = stablehlo.broadcast_in_dim %593, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %595 = stablehlo.concatenate %585, %594, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %596 = mhlo.bitcast %595 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %597 = "stablehlo.gather"(%568, %596) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %598 = mhlo.bitcast %597 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %599 = stablehlo.transpose %598, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %600 = mhlo.copy %599 : tensor<8x256x128x128xf32>
      %601 = stablehlo.subtract %600, %573 : tensor<8x256x128x128xf32>
      %602 = stablehlo.convert %587 : (tensor<128xi32>) -> tensor<128xf32>
      %603 = stablehlo.subtract %577, %602 : tensor<128xf32>
      %604 = stablehlo.clamp %417, %603, %452 : tensor<128xf32>
      %605 = stablehlo.broadcast_in_dim %604, dims = [3] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %606 = stablehlo.multiply %601, %605 : tensor<8x256x128x128xf32>
      %607 = stablehlo.add %573, %606 : tensor<8x256x128x128xf32>
      %608 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %609 = "stablehlo.gather"(%568, %608) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %610 = mhlo.bitcast %609 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %611 = stablehlo.transpose %610, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %612 = mhlo.copy %611 : tensor<8x256x128x128xf32>
      %613 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %614 = stablehlo.add %579, %463 : tensor<128x1xi32>
      %615 = stablehlo.minimum %613, %614 : tensor<128x1xi32>
      %616 = stablehlo.compare LT, %615, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %617 = stablehlo.add %615, %581 : tensor<128x1xi32>
      %618 = stablehlo.select %616, %617, %615 : tensor<128x1xi1>, tensor<128x1xi32>
      %619 = mhlo.bitcast %618 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %620 = stablehlo.broadcast_in_dim %619, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %621 = stablehlo.concatenate %620, %594, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %622 = mhlo.bitcast %621 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %623 = "stablehlo.gather"(%568, %622) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %624 = mhlo.bitcast %623 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %625 = stablehlo.transpose %624, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %626 = mhlo.copy %625 : tensor<8x256x128x128xf32>
      %627 = stablehlo.subtract %626, %612 : tensor<8x256x128x128xf32>
      %628 = stablehlo.multiply %627, %605 : tensor<8x256x128x128xf32>
      %629 = stablehlo.add %612, %628 : tensor<8x256x128x128xf32>
      %630 = stablehlo.subtract %629, %607 : tensor<8x256x128x128xf32>
      %631 = stablehlo.convert %579 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %632 = stablehlo.subtract %578, %631 : tensor<128x1xf32>
      %633 = stablehlo.clamp %481, %632, %484 : tensor<128x1xf32>
      %634 = mhlo.bitcast %633 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %635 = stablehlo.broadcast_in_dim %634, dims = [2] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %636 = stablehlo.multiply %630, %635 : tensor<8x256x128x128xf32>
      %637 = stablehlo.add %607, %636 : tensor<8x256x128x128xf32>
      %638 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<8x16384x256xf32>
      %639 = stablehlo.add %arg210, %638 : tensor<8x16384x256xf32>
      %640 = stablehlo.transpose %639, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[8,256,16384]{1,2,0}"} : (tensor<8x16384x256xf32>) -> tensor<8x256x16384xf32>
      %641 = mhlo.copy %640 : tensor<8x256x16384xf32>
      %642 = mhlo.bitcast %641 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x16384xf32>) -> tensor<8x256x128x128xf32>
      %643 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %644 = "stablehlo.gather"(%642, %643) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %646 = stablehlo.transpose %645, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %647 = mhlo.copy %646 : tensor<8x256x128x128xf32>
      %648 = stablehlo.iota dim = 0 : tensor<128xi32>
      %649 = stablehlo.convert %648 : (tensor<128xi32>) -> tensor<128xf32>
      %650 = stablehlo.maximum %417, %649 : tensor<128xf32>
      %651 = mhlo.bitcast %650 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %652 = stablehlo.convert %651 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %653 = stablehlo.compare LT, %652, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_13 = stablehlo.constant dense<128> : tensor<i32>
      %654 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %655 = stablehlo.add %652, %654 : tensor<128x1xi32>
      %656 = stablehlo.select %653, %655, %652 : tensor<128x1xi1>, tensor<128x1xi32>
      %657 = mhlo.bitcast %656 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %658 = stablehlo.broadcast_in_dim %657, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_14 = stablehlo.constant dense<127> : tensor<i32>
      %659 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %660 = stablehlo.convert %650 : (tensor<128xf32>) -> tensor<128xi32>
      %661 = stablehlo.add %660, %434 : tensor<128xi32>
      %662 = stablehlo.minimum %659, %661 : tensor<128xi32>
      %663 = stablehlo.compare LT, %662, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %664 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %665 = stablehlo.add %662, %664 : tensor<128xi32>
      %666 = stablehlo.select %663, %665, %662 : tensor<128xi1>, tensor<128xi32>
      %667 = stablehlo.broadcast_in_dim %666, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %668 = stablehlo.concatenate %658, %667, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %669 = mhlo.bitcast %668 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %670 = "stablehlo.gather"(%642, %669) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %671 = mhlo.bitcast %670 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %672 = stablehlo.transpose %671, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %673 = mhlo.copy %672 : tensor<8x256x128x128xf32>
      %674 = stablehlo.subtract %673, %647 : tensor<8x256x128x128xf32>
      %675 = stablehlo.convert %660 : (tensor<128xi32>) -> tensor<128xf32>
      %676 = stablehlo.subtract %650, %675 : tensor<128xf32>
      %677 = stablehlo.clamp %417, %676, %452 : tensor<128xf32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [3] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %679 = stablehlo.multiply %674, %678 : tensor<8x256x128x128xf32>
      %680 = stablehlo.add %647, %679 : tensor<8x256x128x128xf32>
      %681 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %682 = "stablehlo.gather"(%642, %681) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %683 = mhlo.bitcast %682 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %684 = stablehlo.transpose %683, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %685 = mhlo.copy %684 : tensor<8x256x128x128xf32>
      %686 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %687 = stablehlo.add %652, %463 : tensor<128x1xi32>
      %688 = stablehlo.minimum %686, %687 : tensor<128x1xi32>
      %689 = stablehlo.compare LT, %688, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %690 = stablehlo.add %688, %654 : tensor<128x1xi32>
      %691 = stablehlo.select %689, %690, %688 : tensor<128x1xi1>, tensor<128x1xi32>
      %692 = mhlo.bitcast %691 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %693 = stablehlo.broadcast_in_dim %692, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %694 = stablehlo.concatenate %693, %667, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %695 = mhlo.bitcast %694 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %696 = "stablehlo.gather"(%642, %695) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x8x256x1x1xf32>
      %697 = mhlo.bitcast %696 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x8x256x1x1xf32>) -> tensor<128x128x8x256xf32>
      %698 = stablehlo.transpose %697, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[8,256,128,128]{1,0,3,2}"} : (tensor<128x128x8x256xf32>) -> tensor<8x256x128x128xf32>
      %699 = mhlo.copy %698 : tensor<8x256x128x128xf32>
      %700 = stablehlo.subtract %699, %685 : tensor<8x256x128x128xf32>
      %701 = stablehlo.multiply %700, %678 : tensor<8x256x128x128xf32>
      %702 = stablehlo.add %685, %701 : tensor<8x256x128x128xf32>
      %703 = stablehlo.subtract %702, %680 : tensor<8x256x128x128xf32>
      %704 = stablehlo.convert %652 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %705 = stablehlo.subtract %651, %704 : tensor<128x1xf32>
      %706 = stablehlo.clamp %481, %705, %484 : tensor<128x1xf32>
      %707 = mhlo.bitcast %706 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %708 = stablehlo.broadcast_in_dim %707, dims = [2] : (tensor<128xf32>) -> tensor<8x256x128x128xf32>
      %709 = stablehlo.multiply %703, %708 : tensor<8x256x128x128xf32>
      %710 = stablehlo.add %680, %709 : tensor<8x256x128x128xf32>
      %711 = stablehlo.concatenate %489, %563, %637, %710, dim = 1 : (tensor<8x256x128x128xf32>, tensor<8x256x128x128xf32>, tensor<8x256x128x128xf32>, tensor<8x256x128x128xf32>) -> tensor<8x1024x128x128xf32>
      %712 = stablehlo.transpose %711, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[8,128,128,1024]{2,1,3,0}"} : (tensor<8x1024x128x128xf32>) -> tensor<8x128x128x1024xf32>
      %713 = mhlo.copy %712 : tensor<8x128x128x1024xf32>
      mhlo.return %713 : tensor<8x128x128x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<8x16384x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<128xf32>, tensor<8x4096x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<8x1024x256xf32>, tensor<256xf32>, tensor<f32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<i32>, tensor<f32>, tensor<8x256x256xf32>, tensor<256xf32>) -> tensor<8x128x128x1024xf32>
    %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x1024xf32>) -> tensor<131072x1024xf32>
    %396 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x1x1xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %408 = mhlo.copy %407 : tensor<1x1x1024x256xf32>
      mhlo.return %408 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %397 = mhlo.bitcast %396 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %398 = stablehlo.dot %395, %397, precision = [DEFAULT, DEFAULT] : (tensor<131072x1024xf32>, tensor<1024x256xf32>) -> tensor<131072x256xf32>
    %399 = mhlo.bitcast %398 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x256xf32>) -> tensor<8x128x128x256xf32>
    %400 = "mhlo.fusion"(%arg203, %arg204, %0, %399, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<8x128x128x256xf32>, %arg212: tensor<256xf32>):
      %407 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<256xf32>) -> tensor<8x128x128x256xf32>
      %408 = stablehlo.subtract %arg211, %407 : tensor<8x128x128x256xf32>
      %409 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<8x128x128x256xf32>
      %410 = stablehlo.multiply %408, %409 : tensor<8x128x128x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<8x128x128x256xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<8x128x128x256xf32>
      %413 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<256xf32>) -> tensor<8x128x128x256xf32>
      %414 = stablehlo.add %412, %413 : tensor<8x128x128x256xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x128x128x256xf32>
      %416 = stablehlo.maximum %414, %415 : tensor<8x128x128x256xf32>
      mhlo.return %416 : tensor<8x128x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x128x128x256xf32>, tensor<256xf32>) -> tensor<8x128x128x256xf32>
    %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x128x256xf32>) -> tensor<131072x256xf32>
    %402 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<150x256x1x1xf32>):
      %407 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,150]{1,0,2,3}"} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
      %408 = mhlo.copy %407 : tensor<1x1x256x150xf32>
      mhlo.return %408 : tensor<1x1x256x150xf32>
    }) {output_operand_aliasing = []} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
    %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x150xf32>) -> tensor<256x150xf32>
    %404 = stablehlo.dot %401, %403, precision = [DEFAULT, DEFAULT] : (tensor<131072x256xf32>, tensor<256x150xf32>) -> tensor<131072x150xf32>
    %405 = mhlo.bitcast %404 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x150xf32>) -> tensor<8x128x128x150xf32>
    %406 = "mhlo.fusion"(%405, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x128x128x150xf32>, %arg209: tensor<150xf32>):
      %407 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<150xf32>) -> tensor<8x128x128x150xf32>
      %408 = stablehlo.add %arg208, %407 : tensor<8x128x128x150xf32>
      %409 = stablehlo.transpose %408, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[8,150,128,128]{1,3,2,0}"} : (tensor<8x128x128x150xf32>) -> tensor<8x150x128x128xf32>
      %410 = mhlo.copy %409 : tensor<8x150x128x128xf32>
      mhlo.return %410 : tensor<8x150x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x128x150xf32>, tensor<150xf32>) -> tensor<8x150x128x128xf32>
    return %406 : tensor<8x150x128x128xf32>
  }
}
