module {
  func.func @main(%arg0: tensor<256xf32>, %arg1: tensor<1x3x512x512xf32>, %arg2: tensor<32x3x7x7xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<32xf32>, %arg7: tensor<32xf32>, %arg8: tensor<32x32xf32>, %arg9: tensor<32xf32>, %arg10: tensor<32x32x8x8xf32>, %arg11: tensor<32xf32>, %arg12: tensor<32xf32>, %arg13: tensor<32xf32>, %arg14: tensor<32x32xf32>, %arg15: tensor<32xf32>, %arg16: tensor<32x32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<128x32xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128x1x3x3xf32>, %arg25: tensor<128xf32>, %arg26: tensor<32x128xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32x32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<32x32x8x8xf32>, %arg33: tensor<32xf32>, %arg34: tensor<32xf32>, %arg35: tensor<32xf32>, %arg36: tensor<32x32xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32x32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<128x32xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x1x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<32x128xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<256x32xf32>, %arg53: tensor<64x32x3x3xf32>, %arg54: tensor<64xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<64xf32>, %arg58: tensor<64xf32>, %arg59: tensor<64x64xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64x64x4x4xf32>, %arg62: tensor<64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<64x64xf32>, %arg66: tensor<64xf32>, %arg67: tensor<64x64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<64x64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64xf32>, %arg73: tensor<256x64xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256x1x3x3xf32>, %arg76: tensor<256xf32>, %arg77: tensor<64x256xf32>, %arg78: tensor<64xf32>, %arg79: tensor<64xf32>, %arg80: tensor<64xf32>, %arg81: tensor<64x64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64x64x4x4xf32>, %arg84: tensor<64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64xf32>, %arg87: tensor<64x64xf32>, %arg88: tensor<64xf32>, %arg89: tensor<64x64xf32>, %arg90: tensor<64xf32>, %arg91: tensor<64x64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64xf32>, %arg95: tensor<256x64xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256x1x3x3xf32>, %arg98: tensor<256xf32>, %arg99: tensor<64x256xf32>, %arg100: tensor<64xf32>, %arg101: tensor<64xf32>, %arg102: tensor<64xf32>, %arg103: tensor<256x64xf32>, %arg104: tensor<160x64x3x3xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160x160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<160x160xf32>, %arg121: tensor<160xf32>, %arg122: tensor<160xf32>, %arg123: tensor<160xf32>, %arg124: tensor<640x160xf32>, %arg125: tensor<640xf32>, %arg126: tensor<640x1x3x3xf32>, %arg127: tensor<640xf32>, %arg128: tensor<160x640xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160x160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<160x160xf32>, %arg143: tensor<160xf32>, %arg144: tensor<160xf32>, %arg145: tensor<160xf32>, %arg146: tensor<640x160xf32>, %arg147: tensor<640xf32>, %arg148: tensor<640x1x3x3xf32>, %arg149: tensor<640xf32>, %arg150: tensor<160x640xf32>, %arg151: tensor<160xf32>, %arg152: tensor<160xf32>, %arg153: tensor<160xf32>, %arg154: tensor<256x160xf32>, %arg155: tensor<256x160x3x3xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<256xf32>, %arg161: tensor<256x256xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256x256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<256x256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256x256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<1024x256xf32>, %arg172: tensor<1024xf32>, %arg173: tensor<1024x1x3x3xf32>, %arg174: tensor<1024xf32>, %arg175: tensor<256x1024xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256x256xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256x256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<256x256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256x256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<1024x256xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<1024x1x3x3xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<256x1024xf32>, %arg194: tensor<256xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256x256xf32>, %arg198: tensor<256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256x1024x1x1xf32>, %arg203: tensor<256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<150x256x1x1xf32>, %arg207: tensor<150xf32>) -> tensor<1x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<256xf32>
      %393 = stablehlo.rsqrt %392 : tensor<256xf32>
      mhlo.return %393 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<127> : tensor<i32>
      %391 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %393 = stablehlo.iota dim = 0 : tensor<128xi32>
      %394 = stablehlo.convert %393 : (tensor<128xi32>) -> tensor<128xf32>
      %395 = stablehlo.maximum %392, %394 : tensor<128xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %397 = stablehlo.convert %396 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_4 = stablehlo.constant dense<1> : tensor<i32>
      %398 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %399 = stablehlo.add %397, %398 : tensor<128x1xi32>
      %400 = stablehlo.minimum %391, %399 : tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %401 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %402 = stablehlo.compare LT, %400, %401 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<128> : tensor<i32>
      %403 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %404 = stablehlo.add %400, %403 : tensor<128x1xi32>
      %405 = stablehlo.select %402, %404, %400 : tensor<128x1xi1>, tensor<128x1xi32>
      %406 = mhlo.bitcast %405 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %407 = stablehlo.broadcast_in_dim %406, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %408 = stablehlo.convert %395 : (tensor<128xf32>) -> tensor<128xi32>
      %409 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %411 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %412 = stablehlo.add %408, %411 : tensor<128xi32>
      %413 = stablehlo.select %410, %412, %408 : tensor<128xi1>, tensor<128xi32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %415 = stablehlo.concatenate %407, %414, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %415 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %392 = stablehlo.iota dim = 0 : tensor<128xi32>
      %393 = stablehlo.convert %392 : (tensor<128xi32>) -> tensor<128xf32>
      %394 = stablehlo.maximum %391, %393 : tensor<128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %396 = stablehlo.convert %395 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_3 = stablehlo.constant dense<0> : tensor<i32>
      %397 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %398 = stablehlo.compare LT, %396, %397 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_4 = stablehlo.constant dense<128> : tensor<i32>
      %399 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %400 = stablehlo.add %396, %399 : tensor<128x1xi32>
      %401 = stablehlo.select %398, %400, %396 : tensor<128x1xi1>, tensor<128x1xi32>
      %402 = mhlo.bitcast %401 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %404 = stablehlo.convert %394 : (tensor<128xf32>) -> tensor<128xi32>
      %405 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %406 = stablehlo.compare LT, %404, %405 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %407 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %408 = stablehlo.add %404, %407 : tensor<128xi32>
      %409 = stablehlo.select %406, %408, %404 : tensor<128xi1>, tensor<128xi32>
      %410 = stablehlo.broadcast_in_dim %409, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %411 = stablehlo.concatenate %403, %410, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %411 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %3 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x3x512x512xf32>):
      %391 = stablehlo.transpose %arg208, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,512,3]{2,1,3,0}"} : (tensor<1x3x512x512xf32>) -> tensor<1x512x512x3xf32>
      %392 = mhlo.copy %391 : tensor<1x512x512x3xf32>
      mhlo.return %392 : tensor<1x512x512x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x512x512xf32>) -> tensor<1x512x512x3xf32>
    %4 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x3x7x7xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,32]{1,0,2,3}"} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
      %392 = mhlo.copy %391 : tensor<7x7x3x32xf32>
      mhlo.return %392 : tensor<7x7x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
    %5 = stablehlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x512x512x3xf32>, tensor<7x7x3x32xf32>) -> tensor<1x128x128x32xf32>
    %6 = "mhlo.fusion"(%5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      mhlo.return %397 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = "mhlo.fusion"(%6, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %8 = "mhlo.fusion"(%7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384xf32>, %arg209: tensor<1x128x128x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x16384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x16384x32xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.reduce(%403 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %405 = stablehlo.multiply %404, %398 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %407 = stablehlo.add %405, %406 : tensor<1x16384xf32>
      %408 = stablehlo.rsqrt %407 : tensor<1x16384xf32>
      mhlo.return %408 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x16384xf32>
    %9 = "mhlo.fusion"(%arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384xf32>, %arg212: tensor<1x128x128x32xf32>, %arg213: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x16384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x16384x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x16384x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x16384x32xf32>
      mhlo.return %409 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %10 = "mhlo.fusion"(%9, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %11 = "mhlo.fusion"(%10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<1x16384xf32>, %arg212: tensor<1x16384xf32>, %arg213: tensor<1x128x128x32xf32>, %arg214: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg213, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %399 = stablehlo.multiply %arg212, %398 : tensor<1x16384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x16384x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x16384x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x16384x32xf32>
      %410 = stablehlo.multiply %arg208, %398 : tensor<1x16384xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x16384x32xf32>
      %414 = stablehlo.multiply %413, %413 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %415 = stablehlo.reduce(%414 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %416 = stablehlo.multiply %415, %398 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x16384xf32>
      %419 = stablehlo.rsqrt %418 : tensor<1x16384xf32>
      mhlo.return %419 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x16384xf32>
    %12 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<1x16384xf32>, %arg215: tensor<1x16384xf32>, %arg216: tensor<1x128x128x32xf32>, %arg217: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x16384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x16384x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x16384x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x16384x32xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x16384xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x16384x32xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x16384x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %421 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<16384x32xf32>
    %13 = stablehlo.dot_general %12, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %14 = "mhlo.fusion"(%13, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<16384x32xf32>
      mhlo.return %392 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<16384x32xf32>
    %15 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<1x16384xf32>, %arg215: tensor<1x16384xf32>, %arg216: tensor<1x128x128x32xf32>, %arg217: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x128x128x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x32x128x128xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %397 = mhlo.copy %396 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x16384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x16384x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x16384x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x16384x32xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x16384xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x16384x32xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x16384x32xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,16384]{1,2,0}"} : (tensor<1x16384x32xf32>) -> tensor<1x32x16384xf32>
      %422 = mhlo.copy %421 : tensor<1x32x16384xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,32]{2,1,3,0}"} : (tensor<1x32x128x128xf32>) -> tensor<1x128x128x32xf32>
      %425 = mhlo.copy %424 : tensor<1x128x128x32xf32>
      mhlo.return %425 : tensor<1x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x128x128x32xf32>
    %16 = "mhlo.fusion"(%arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %392 = mhlo.copy %391 : tensor<8x8x32x32xf32>
      mhlo.return %392 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %17 = stablehlo.convolution(%15, %16) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<1x16x16x32xf32>
    %18 = "mhlo.fusion"(%17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %398 = stablehlo.reduce(%397 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %398 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<1x256xf32>
    %19 = "mhlo.fusion"(%18, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x32xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.reduce(%403 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
      %405 = stablehlo.multiply %404, %398 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %407 = stablehlo.add %405, %406 : tensor<1x256xf32>
      %408 = stablehlo.rsqrt %407 : tensor<1x256xf32>
      mhlo.return %408 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<1x256xf32>
    %20 = "mhlo.fusion"(%arg12, %arg13, %19, %18, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x32xf32>, %arg213: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x32xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x32xf32>) -> tensor<256x32xf32>
      mhlo.return %410 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
    %21 = stablehlo.dot_general %20, %arg14, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %22 = "mhlo.fusion"(%21, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x32xf32>) -> tensor<1x1x256x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,1,32,256]{2,3,1,0}"} : (tensor<1x1x256x32xf32>) -> tensor<1x1x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x1x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x32x256xf32>) -> tensor<32x256xf32>
      mhlo.return %396 : tensor<32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>) -> tensor<32x256xf32>
    %23 = stablehlo.dot %14, %22, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %24 = "mhlo.fusion"(%23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<16384x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
      mhlo.return %393 : tensor<1x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %25 = "mhlo.fusion"(%24, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384x8xf32>
      mhlo.return %391 : tensor<1x1x16384x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384x8xf32>
    %26 = "mhlo.fusion"(%25, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1x16384x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x16384x8xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
      mhlo.return %391 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x8xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %27 = "mhlo.fusion"(%26, %23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384xf32>, %arg209: tensor<16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<16384x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x1x16384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [2] : (tensor<16384xf32>) -> tensor<1x1x16384x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x1x16384x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x1x16384x256xf32>
      mhlo.return %399 : tensor<1x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %28 = "mhlo.fusion"(%27, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
      mhlo.return %391 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %29 = "mhlo.fusion"(%27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<1x1x16384xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<16384xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [2] : (tensor<16384xf32>) -> tensor<1x1x16384x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x1x16384x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x16384x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %394 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<1x1x16384xf32>) -> tensor<16384x256xf32>
    %30 = stablehlo.dot_general %20, %arg16, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %31 = "mhlo.fusion"(%30, %arg17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x32xf32>
      mhlo.return %392 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
    %32 = stablehlo.dot %29, %31, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    %33 = stablehlo.dot_general %32, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %34 = "mhlo.fusion"(%33, %arg4, %arg5, %8, %7, %5, %arg3, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<1x16384xf32>, %arg212: tensor<1x16384xf32>, %arg213: tensor<1x128x128x32xf32>, %arg214: tensor<32xf32>, %arg215: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<16384x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<1x16384x32xf32>
      %394 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %395 = stablehlo.add %arg213, %394 : tensor<1x128x128x32xf32>
      %396 = stablehlo.transpose %395, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %397 = mhlo.copy %396 : tensor<1x32x128x128xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %399 = stablehlo.transpose %398, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %400 = mhlo.copy %399 : tensor<1x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %402 = stablehlo.multiply %arg212, %401 : tensor<1x16384xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %405 = stablehlo.subtract %400, %404 : tensor<1x16384x32xf32>
      %406 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %407 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %408 = stablehlo.multiply %405, %407 : tensor<1x16384x32xf32>
      %409 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %410 = stablehlo.multiply %408, %409 : tensor<1x16384x32xf32>
      %411 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %412 = stablehlo.add %410, %411 : tensor<1x16384x32xf32>
      %413 = stablehlo.add %393, %412 : tensor<1x16384x32xf32>
      mhlo.return %413 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %36 = "mhlo.fusion"(%34, %35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x16384x32xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %397 = stablehlo.reduce(%396 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %398 = stablehlo.multiply %397, %391 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %400 = stablehlo.add %398, %399 : tensor<1x16384xf32>
      %401 = stablehlo.rsqrt %400 : tensor<1x16384xf32>
      mhlo.return %401 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x16384xf32>
    %37 = "mhlo.fusion"(%arg20, %arg21, %36, %34, %35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %403 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<16384x32xf32>
    %38 = stablehlo.dot_general %37, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    %39 = "mhlo.fusion"(%38, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<128xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<16384x128xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<16384x128xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<1x16384x128xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,128,16384]{1,2,0}"} : (tensor<1x16384x128xf32>) -> tensor<1x128x16384xf32>
      %395 = mhlo.copy %394 : tensor<1x128x16384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x16384xf32>) -> tensor<1x128x128x128xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{2,1,3,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %398 = mhlo.copy %397 : tensor<1x128x128x128xf32>
      mhlo.return %398 : tensor<1x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %40 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x128xf32>
      mhlo.return %392 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %41 = stablehlo.convolution(%39, %40) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<1x128x128x128xf32>
    %42 = "mhlo.fusion"(%41, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x128x128x128xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x128x128x128xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{1,3,2,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %396 = mhlo.copy %395 : tensor<1x128x128x128xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,128]{1,2,0}"} : (tensor<1x128x16384xf32>) -> tensor<1x16384x128xf32>
      %399 = mhlo.copy %398 : tensor<1x16384x128xf32>
      %400 = stablehlo.negate %393 : tensor<1x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x128x128x128xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{1,3,2,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %404 = mhlo.copy %403 : tensor<1x128x128x128xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,128]{1,2,0}"} : (tensor<1x128x16384xf32>) -> tensor<1x16384x128xf32>
      %407 = mhlo.copy %406 : tensor<1x16384x128xf32>
      %408 = stablehlo.abs %407 : tensor<1x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x16384x128xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x16384x128xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x16384x128xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x16384x128xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x16384x128xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x16384x128xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x16384x128xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x16384x128xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %436 = stablehlo.negate %412 : tensor<1x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %439 = stablehlo.exponential %436 : tensor<1x16384x128xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x16384x128xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x16384x128xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x16384x128xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x16384x128xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x16384x128xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x16384x128xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x16384x128xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x16384x128xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x16384x128xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x16384x128xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x16384x128xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x16384x128xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x16384x128xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x16384x128xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x16384x128xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x16384x128xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x16384x128xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x16384x128xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x16384x128xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x16384x128xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x128xf32>) -> tensor<16384x128xf32>
      mhlo.return %498 : tensor<16384x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x128xf32>, tensor<128xf32>) -> tensor<16384x128xf32>
    %43 = stablehlo.dot_general %42, %arg26, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    %44 = "mhlo.fusion"(%34, %43, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<16384x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<1x16384x32xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x16384x32xf32>
      mhlo.return %394 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %45 = "mhlo.fusion"(%44, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %46 = "mhlo.fusion"(%44, %45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x16384x32xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %397 = stablehlo.reduce(%396 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %398 = stablehlo.multiply %397, %391 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %400 = stablehlo.add %398, %399 : tensor<1x16384xf32>
      %401 = stablehlo.rsqrt %400 : tensor<1x16384xf32>
      mhlo.return %401 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x16384xf32>
    %47 = "mhlo.fusion"(%arg28, %arg29, %46, %44, %45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %403 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<16384x32xf32>
    %48 = stablehlo.dot_general %47, %arg30, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %49 = "mhlo.fusion"(%48, %arg31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<16384x32xf32>
      mhlo.return %392 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<16384x32xf32>
    %50 = "mhlo.fusion"(%arg28, %arg29, %46, %44, %45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = stablehlo.transpose %402, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,16384]{1,2,0}"} : (tensor<1x16384x32xf32>) -> tensor<1x32x16384xf32>
      %404 = mhlo.copy %403 : tensor<1x32x16384xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x32x16384xf32>) -> tensor<1x32x128x128xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,32]{2,1,3,0}"} : (tensor<1x32x128x128xf32>) -> tensor<1x128x128x32xf32>
      %407 = mhlo.copy %406 : tensor<1x128x128x32xf32>
      mhlo.return %407 : tensor<1x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x128x128x32xf32>
    %51 = "mhlo.fusion"(%arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %392 = mhlo.copy %391 : tensor<8x8x32x32xf32>
      mhlo.return %392 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %52 = stablehlo.convolution(%50, %51) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<1x16x16x32xf32>
    %53 = "mhlo.fusion"(%52, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %398 = stablehlo.reduce(%397 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %398 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<1x256xf32>
    %54 = "mhlo.fusion"(%53, %52, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x32xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.reduce(%403 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x256x32xf32>, tensor<f32>) -> tensor<1x256xf32>
      %405 = stablehlo.multiply %404, %398 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %406 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %407 = stablehlo.add %405, %406 : tensor<1x256xf32>
      %408 = stablehlo.rsqrt %407 : tensor<1x256xf32>
      mhlo.return %408 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<1x256xf32>
    %55 = "mhlo.fusion"(%arg34, %arg35, %54, %53, %52, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x32xf32>, %arg213: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<1x16x16x32xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x32xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,16,16]{1,3,2,0}"} : (tensor<1x16x16x32xf32>) -> tensor<1x32x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x32x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x16x16xf32>) -> tensor<1x32x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,32]{1,2,0}"} : (tensor<1x32x256xf32>) -> tensor<1x256x32xf32>
      %397 = mhlo.copy %396 : tensor<1x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x32xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x32xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x32xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x32xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x256x32xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x32xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x32xf32>) -> tensor<256x32xf32>
      mhlo.return %410 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
    %56 = stablehlo.dot_general %55, %arg36, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %57 = "mhlo.fusion"(%56, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x32xf32>) -> tensor<1x1x256x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,1,32,256]{2,3,1,0}"} : (tensor<1x1x256x32xf32>) -> tensor<1x1x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x1x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x32x256xf32>) -> tensor<32x256xf32>
      mhlo.return %396 : tensor<32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>) -> tensor<32x256xf32>
    %58 = stablehlo.dot %49, %57, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x256xf32>) -> tensor<16384x256xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<16384x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
      mhlo.return %393 : tensor<1x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %60 = "mhlo.fusion"(%59, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384x8xf32>
      mhlo.return %391 : tensor<1x1x16384x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384x8xf32>
    %61 = "mhlo.fusion"(%60, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1x16384x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x16384x8xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
      mhlo.return %391 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x8xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %62 = "mhlo.fusion"(%61, %58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384xf32>, %arg209: tensor<16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16384x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<16384x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x1x16384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [2] : (tensor<16384xf32>) -> tensor<1x1x16384x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x1x16384x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x1x16384x256xf32>
      mhlo.return %399 : tensor<1x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<16384x256xf32>) -> tensor<1x1x16384x256xf32>
    %63 = "mhlo.fusion"(%62, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
      mhlo.return %391 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<f32>) -> tensor<1x1x16384xf32>
    %64 = "mhlo.fusion"(%62, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1x16384x256xf32>, %arg209: tensor<1x1x16384xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<16384xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [2] : (tensor<16384xf32>) -> tensor<1x1x16384x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x1x16384x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x16384x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %394 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384x256xf32>, tensor<1x1x16384xf32>) -> tensor<16384x256xf32>
    %65 = stablehlo.dot_general %55, %arg38, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x32xf32>, tensor<32x32xf32>) -> tensor<256x32xf32>
    %66 = "mhlo.fusion"(%65, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>, %arg209: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<256x32xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x32xf32>
      mhlo.return %392 : tensor<256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
    %67 = stablehlo.dot %64, %66, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x32xf32>) -> tensor<16384x32xf32>
    %68 = stablehlo.dot_general %67, %arg40, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %69 = "mhlo.fusion"(%44, %68, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<16384x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<1x16384x32xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x16384x32xf32>
      mhlo.return %394 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %70 = "mhlo.fusion"(%69, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %71 = "mhlo.fusion"(%69, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x16384x32xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %397 = stablehlo.reduce(%396 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %398 = stablehlo.multiply %397, %391 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %400 = stablehlo.add %398, %399 : tensor<1x16384xf32>
      %401 = stablehlo.rsqrt %400 : tensor<1x16384xf32>
      mhlo.return %401 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x16384xf32>
    %72 = "mhlo.fusion"(%arg42, %arg43, %71, %69, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %403 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<16384x32xf32>
    %73 = stablehlo.dot_general %72, %arg44, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<128x32xf32>) -> tensor<16384x128xf32>
    %74 = "mhlo.fusion"(%73, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x128xf32>, %arg209: tensor<128xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<16384x128xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<16384x128xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<1x16384x128xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,128,16384]{1,2,0}"} : (tensor<1x16384x128xf32>) -> tensor<1x128x16384xf32>
      %395 = mhlo.copy %394 : tensor<1x128x16384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x16384xf32>) -> tensor<1x128x128x128xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{2,1,3,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %398 = mhlo.copy %397 : tensor<1x128x128x128xf32>
      mhlo.return %398 : tensor<1x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x128xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %75 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x128xf32>
      mhlo.return %392 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %76 = stablehlo.convolution(%74, %75) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<1x128x128x128xf32>
    %77 = "mhlo.fusion"(%76, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x128x128x128xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x128x128x128xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{1,3,2,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %396 = mhlo.copy %395 : tensor<1x128x128x128xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,128]{1,2,0}"} : (tensor<1x128x16384xf32>) -> tensor<1x16384x128xf32>
      %399 = mhlo.copy %398 : tensor<1x16384x128xf32>
      %400 = stablehlo.negate %393 : tensor<1x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x128x128x128xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{1,3,2,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %404 = mhlo.copy %403 : tensor<1x128x128x128xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x128xf32>) -> tensor<1x128x16384xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,128]{1,2,0}"} : (tensor<1x128x16384xf32>) -> tensor<1x16384x128xf32>
      %407 = mhlo.copy %406 : tensor<1x16384x128xf32>
      %408 = stablehlo.abs %407 : tensor<1x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x16384x128xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x16384x128xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x16384x128xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x16384x128xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x16384x128xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x16384x128xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x16384x128xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x16384x128xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %436 = stablehlo.negate %412 : tensor<1x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %439 = stablehlo.exponential %436 : tensor<1x16384x128xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x16384x128xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x16384x128xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x16384x128xf32>, tensor<1x16384x128xf32>) -> tensor<1x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x16384x128xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x16384x128xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x16384x128xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x16384x128xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x16384x128xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x16384x128xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x16384x128xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x16384x128xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x16384x128xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x16384x128xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x16384x128xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x16384x128xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x16384x128xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x16384x128xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x16384x128xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x16384x128xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x16384x128xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x16384x128xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x16384x128xi1>, tensor<1x16384x128xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x16384x128xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x128xf32>) -> tensor<16384x128xf32>
      mhlo.return %498 : tensor<16384x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x128xf32>, tensor<128xf32>) -> tensor<16384x128xf32>
    %78 = stablehlo.dot_general %77, %arg48, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x128xf32>, tensor<32x128xf32>) -> tensor<16384x32xf32>
    %79 = "mhlo.fusion"(%69, %78, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<16384x32xf32>, %arg210: tensor<32xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<16384x32xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<1x16384x32xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x16384x32xf32>
      mhlo.return %394 : tensor<1x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<16384x32xf32>, tensor<32xf32>) -> tensor<1x16384x32xf32>
    %80 = "mhlo.fusion"(%79, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      mhlo.return %391 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
    %81 = "mhlo.fusion"(%79, %80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16384x32xf32>, %arg209: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x16384x32xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %397 = stablehlo.reduce(%396 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<1x16384x32xf32>, tensor<f32>) -> tensor<1x16384xf32>
      %398 = stablehlo.multiply %397, %391 : tensor<1x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %400 = stablehlo.add %398, %399 : tensor<1x16384xf32>
      %401 = stablehlo.rsqrt %400 : tensor<1x16384xf32>
      mhlo.return %401 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x16384xf32>
    %82 = "mhlo.fusion"(%arg50, %arg51, %81, %79, %80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<1x128x128x32xf32>
      %404 = stablehlo.transpose %403, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %405 = mhlo.copy %404 : tensor<1x32x128x128xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x128x128xf32>) -> tensor<1x32x16384xf32>
      %407 = stablehlo.transpose %406, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,16384,32]{1,2,0}"} : (tensor<1x32x16384xf32>) -> tensor<1x16384x32xf32>
      %408 = mhlo.copy %407 : tensor<1x16384x32xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %409 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<16384x32xf32>
    %83 = stablehlo.dot_general %82, %arg52, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<256x32xf32>) -> tensor<16384x256xf32>
    %84 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<63> : tensor<i32>
      %391 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %393 = stablehlo.iota dim = 0 : tensor<128xi32>
      %394 = stablehlo.convert %393 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %396 = stablehlo.add %394, %395 : tensor<128xf32>
      %397 = stablehlo.multiply %396, %395 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %399 = stablehlo.add %397, %398 : tensor<128xf32>
      %400 = stablehlo.maximum %392, %399 : tensor<128xf32>
      %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %402 = stablehlo.convert %401 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<1> : tensor<i32>
      %403 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %404 = stablehlo.add %402, %403 : tensor<128x1xi32>
      %405 = stablehlo.minimum %391, %404 : tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<0> : tensor<i32>
      %406 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %407 = stablehlo.compare LT, %405, %406 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<64> : tensor<i32>
      %408 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %409 = stablehlo.add %405, %408 : tensor<128x1xi32>
      %410 = stablehlo.select %407, %409, %405 : tensor<128x1xi1>, tensor<128x1xi32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %413 = stablehlo.convert %400 : (tensor<128xf32>) -> tensor<128xi32>
      %414 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %415 = stablehlo.compare LT, %413, %414 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %416 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %417 = stablehlo.add %413, %416 : tensor<128xi32>
      %418 = stablehlo.select %415, %417, %413 : tensor<128xi1>, tensor<128xi32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %420 = stablehlo.concatenate %412, %419, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %420 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %85 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %392 = stablehlo.iota dim = 0 : tensor<128xi32>
      %393 = stablehlo.convert %392 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %395 = stablehlo.add %393, %394 : tensor<128xf32>
      %396 = stablehlo.multiply %395, %394 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %398 = stablehlo.add %396, %397 : tensor<128xf32>
      %399 = stablehlo.maximum %391, %398 : tensor<128xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %401 = stablehlo.convert %400 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %402 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %403 = stablehlo.compare LT, %401, %402 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<64> : tensor<i32>
      %404 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %405 = stablehlo.add %401, %404 : tensor<128x1xi32>
      %406 = stablehlo.select %403, %405, %401 : tensor<128x1xi1>, tensor<128x1xi32>
      %407 = mhlo.bitcast %406 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %408 = stablehlo.broadcast_in_dim %407, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %409 = stablehlo.convert %399 : (tensor<128xf32>) -> tensor<128xi32>
      %410 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %411 = stablehlo.compare LT, %409, %410 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %412 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %413 = stablehlo.add %409, %412 : tensor<128xi32>
      %414 = stablehlo.select %411, %413, %409 : tensor<128xi1>, tensor<128xi32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %416 = stablehlo.concatenate %408, %415, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %416 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %86 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %391 = stablehlo.iota dim = 0 : tensor<128xi32>
      %392 = stablehlo.convert %391 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %393 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %394 = stablehlo.add %392, %393 : tensor<128xf32>
      mhlo.return %394 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %87 = "mhlo.fusion"(%arg50, %arg51, %81, %79, %80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<1x16384xf32>, %arg211: tensor<1x16384x32xf32>, %arg212: tensor<1x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16384xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x16384xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x16384x32xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x16384xf32>) -> tensor<16384xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<16384xf32>) -> tensor<1x16384x32xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x16384x32xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x16384x32xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<1x16384x32xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x16384x32xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16384x32xf32>) -> tensor<1x128x128x32xf32>
      mhlo.return %403 : tensor<1x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<1x16384xf32>, tensor<1x16384x32xf32>, tensor<1x16384xf32>) -> tensor<1x128x128x32xf32>
    %88 = "mhlo.fusion"(%arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %392 = mhlo.copy %391 : tensor<3x3x32x64xf32>
      mhlo.return %392 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %89 = stablehlo.convolution(%87, %88) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x32xf32>, tensor<3x3x32x64xf32>) -> tensor<1x64x64x64xf32>
    %90 = "mhlo.fusion"(%89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x64x64x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      mhlo.return %397 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %91 = "mhlo.fusion"(%90, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %92 = "mhlo.fusion"(%91, %89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096xf32>, %arg209: tensor<1x64x64x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x4096xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x4096x64xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x4096x64xf32>
      mhlo.return %403 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %93 = "mhlo.fusion"(%92, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %95 = "mhlo.fusion"(%arg55, %arg56, %94, %91, %89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096xf32>, %arg212: tensor<1x64x64x64xf32>, %arg213: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x4096xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x4096x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x4096x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x4096x64xf32>
      mhlo.return %409 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %96 = "mhlo.fusion"(%95, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %97 = "mhlo.fusion"(%96, %arg55, %arg56, %94, %91, %89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<1x4096xf32>, %arg212: tensor<1x4096xf32>, %arg213: tensor<1x64x64x64xf32>, %arg214: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg213, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %399 = stablehlo.multiply %arg212, %398 : tensor<1x4096xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x4096x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x4096x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x4096x64xf32>
      %410 = stablehlo.multiply %arg208, %398 : tensor<1x4096xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x4096x64xf32>
      %414 = stablehlo.multiply %413, %413 : tensor<1x4096x64xf32>
      mhlo.return %414 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %98 = "mhlo.fusion"(%97, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %99 = "mhlo.fusion"(%98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %100 = "mhlo.fusion"(%arg57, %arg58, %99, %96, %arg55, %arg56, %94, %91, %89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<1x4096xf32>, %arg215: tensor<1x4096xf32>, %arg216: tensor<1x64x64x64xf32>, %arg217: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x4096xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x4096x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x4096x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x4096x64xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x4096xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x4096x64xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x4096x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %421 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<4096x64xf32>
    %101 = stablehlo.dot_general %100, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %102 = "mhlo.fusion"(%101, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,4096,32]{3,1,2,0}"} : (tensor<1x4096x2x32xf32>) -> tensor<1x2x4096x32xf32>
      %395 = mhlo.copy %394 : tensor<1x2x4096x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x4096x32xf32>) -> tensor<2x4096x32xf32>
      mhlo.return %396 : tensor<2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<2x4096x32xf32>
    %103 = "mhlo.fusion"(%arg57, %arg58, %99, %96, %arg55, %arg56, %94, %91, %89, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<1x4096xf32>, %arg215: tensor<1x4096xf32>, %arg216: tensor<1x64x64x64xf32>, %arg217: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x64x64x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %394 = mhlo.copy %393 : tensor<1x64x64x64xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %397 = mhlo.copy %396 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x4096xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x4096x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x4096x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x4096x64xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x4096xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x4096x64xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x4096x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,64,4096]{1,2,0}"} : (tensor<1x4096x64xf32>) -> tensor<1x64x4096xf32>
      %422 = mhlo.copy %421 : tensor<1x64x4096xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{2,1,3,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %425 = mhlo.copy %424 : tensor<1x64x64x64xf32>
      mhlo.return %425 : tensor<1x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>) -> tensor<1x64x64x64xf32>
    %104 = "mhlo.fusion"(%arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %392 = mhlo.copy %391 : tensor<4x4x64x64xf32>
      mhlo.return %392 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %105 = stablehlo.convolution(%103, %104) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<1x16x16x64xf32>
    %106 = "mhlo.fusion"(%105, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      mhlo.return %397 : tensor<1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<1x256x64xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
      mhlo.return %391 : tensor<1x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x2xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
    %109 = "mhlo.fusion"(%108, %105, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x64xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x64xf32>
      mhlo.return %403 : tensor<1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<1x256x64xf32>
    %110 = "mhlo.fusion"(%109, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
      mhlo.return %391 : tensor<1x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2xf32>) -> tensor<1x256xf32>
    %112 = "mhlo.fusion"(%arg63, %arg64, %111, %108, %105, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x64xf32>, %arg213: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x64xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x64xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x64xf32>) -> tensor<256x64xf32>
      mhlo.return %410 : tensor<256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<256x64xf32>
    %113 = stablehlo.dot_general %112, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %114 = "mhlo.fusion"(%113, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x64xf32>) -> tensor<1x256x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,32,256]{2,1,3,0}"} : (tensor<1x256x2x32xf32>) -> tensor<1x2x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x2x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x32x256xf32>) -> tensor<2x32x256xf32>
      mhlo.return %396 : tensor<2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>, tensor<64xf32>) -> tensor<2x32x256xf32>
    %115 = stablehlo.dot_general %102, %114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<2x4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
      mhlo.return %393 : tensor<1x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %117 = "mhlo.fusion"(%116, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096x8xf32>
      mhlo.return %391 : tensor<1x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096x8xf32>
    %118 = "mhlo.fusion"(%117, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x2x4096x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4096x8xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
      mhlo.return %391 : tensor<1x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x8xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %119 = "mhlo.fusion"(%118, %115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096xf32>, %arg209: tensor<2x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<2x4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x2x4096xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x2x4096xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x2x4096xf32>) -> tensor<2x4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<2x4096xf32>) -> tensor<1x2x4096x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x2x4096x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x2x4096x256xf32>
      mhlo.return %399 : tensor<1x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096xf32>, tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %120 = "mhlo.fusion"(%119, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
      mhlo.return %391 : tensor<1x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %121 = "mhlo.fusion"(%119, %120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<1x2x4096xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x2x4096xf32>) -> tensor<2x4096xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<2x4096xf32>) -> tensor<1x2x4096x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x2x4096x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x4096x256xf32>) -> tensor<2x4096x256xf32>
      mhlo.return %394 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<1x2x4096xf32>) -> tensor<2x4096x256xf32>
    %122 = stablehlo.dot_general %112, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %123 = "mhlo.fusion"(%122, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x64xf32>) -> tensor<1x256x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,256,32]{3,1,2,0}"} : (tensor<1x256x2x32xf32>) -> tensor<1x2x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x2x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x256x32xf32>) -> tensor<2x256x32xf32>
      mhlo.return %396 : tensor<2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>, tensor<64xf32>) -> tensor<2x256x32xf32>
    %124 = stablehlo.dot_general %121, %123, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
    %125 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2x4096x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x32xf32>) -> tensor<1x2x4096x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4096,2,32]{3,1,2,0}"} : (tensor<1x2x4096x32xf32>) -> tensor<1x4096x2x32xf32>
      %393 = mhlo.copy %392 : tensor<1x4096x2x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4096x2x32xf32>) -> tensor<4096x64xf32>
      mhlo.return %394 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x32xf32>) -> tensor<4096x64xf32>
    %126 = stablehlo.dot_general %125, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %127 = "mhlo.fusion"(%126, %arg55, %arg56, %94, %91, %89, %arg54, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<1x4096xf32>, %arg212: tensor<1x4096xf32>, %arg213: tensor<1x64x64x64xf32>, %arg214: tensor<64xf32>, %arg215: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x64xf32>
      %394 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<1x64x64x64xf32>
      %395 = stablehlo.add %arg213, %394 : tensor<1x64x64x64xf32>
      %396 = stablehlo.transpose %395, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %397 = mhlo.copy %396 : tensor<1x64x64x64xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %399 = stablehlo.transpose %398, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %400 = mhlo.copy %399 : tensor<1x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %402 = stablehlo.multiply %arg212, %401 : tensor<1x4096xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %405 = stablehlo.subtract %400, %404 : tensor<1x4096x64xf32>
      %406 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %407 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %408 = stablehlo.multiply %405, %407 : tensor<1x4096x64xf32>
      %409 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %410 = stablehlo.multiply %408, %409 : tensor<1x4096x64xf32>
      %411 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %412 = stablehlo.add %410, %411 : tensor<1x4096x64xf32>
      %413 = stablehlo.add %393, %412 : tensor<1x4096x64xf32>
      mhlo.return %413 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096xf32>, tensor<1x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %128 = "mhlo.fusion"(%127, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %129 = "mhlo.fusion"(%127, %128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x4096x64xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x4096x64xf32>
      mhlo.return %396 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x4096x64xf32>
    %130 = "mhlo.fusion"(%129, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %131 = "mhlo.fusion"(%130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %132 = "mhlo.fusion"(%arg71, %arg72, %131, %127, %128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %403 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<4096x64xf32>
    %133 = stablehlo.dot_general %132, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %134 = "mhlo.fusion"(%133, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<1x4096x256xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,4096]{1,2,0}"} : (tensor<1x4096x256xf32>) -> tensor<1x256x4096xf32>
      %395 = mhlo.copy %394 : tensor<1x256x4096xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4096xf32>) -> tensor<1x256x64x64xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,256]{2,1,3,0}"} : (tensor<1x256x64x64xf32>) -> tensor<1x64x64x256xf32>
      %398 = mhlo.copy %397 : tensor<1x64x64x256xf32>
      mhlo.return %398 : tensor<1x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<1x64x64x256xf32>
    %135 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x256xf32>
      mhlo.return %392 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %136 = stablehlo.convolution(%134, %135) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<1x64x64x256xf32>
    %137 = "mhlo.fusion"(%136, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x64x256xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x64x64x256xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x64x64x256xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %396 = mhlo.copy %395 : tensor<1x256x64x64xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,256]{1,2,0}"} : (tensor<1x256x4096xf32>) -> tensor<1x4096x256xf32>
      %399 = mhlo.copy %398 : tensor<1x4096x256xf32>
      %400 = stablehlo.negate %393 : tensor<1x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x64x256xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x64x64x256xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %404 = mhlo.copy %403 : tensor<1x256x64x64xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,256]{1,2,0}"} : (tensor<1x256x4096xf32>) -> tensor<1x4096x256xf32>
      %407 = mhlo.copy %406 : tensor<1x4096x256xf32>
      %408 = stablehlo.abs %407 : tensor<1x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x4096x256xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x4096x256xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x4096x256xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x4096x256xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x4096x256xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x4096x256xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x4096x256xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x4096x256xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %436 = stablehlo.negate %412 : tensor<1x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %439 = stablehlo.exponential %436 : tensor<1x4096x256xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x4096x256xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x4096x256xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x4096x256xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x4096x256xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x4096x256xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x4096x256xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x4096x256xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x4096x256xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x4096x256xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x4096x256xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x4096x256xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x4096x256xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x4096x256xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x4096x256xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x4096x256xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x4096x256xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x4096x256xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x4096x256xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x4096x256xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x4096x256xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %498 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x256xf32>, tensor<256xf32>) -> tensor<4096x256xf32>
    %138 = stablehlo.dot_general %137, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    %139 = "mhlo.fusion"(%127, %138, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x64xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x4096x64xf32>
      mhlo.return %394 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %140 = "mhlo.fusion"(%139, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %141 = "mhlo.fusion"(%139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x4096x64xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x4096x64xf32>
      mhlo.return %396 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x4096x64xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %144 = "mhlo.fusion"(%arg79, %arg80, %143, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %403 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<4096x64xf32>
    %145 = stablehlo.dot_general %144, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %146 = "mhlo.fusion"(%145, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,4096,32]{3,1,2,0}"} : (tensor<1x4096x2x32xf32>) -> tensor<1x2x4096x32xf32>
      %395 = mhlo.copy %394 : tensor<1x2x4096x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x4096x32xf32>) -> tensor<2x4096x32xf32>
      mhlo.return %396 : tensor<2x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<2x4096x32xf32>
    %147 = "mhlo.fusion"(%arg79, %arg80, %143, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = stablehlo.transpose %402, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,64,4096]{1,2,0}"} : (tensor<1x4096x64xf32>) -> tensor<1x64x4096xf32>
      %404 = mhlo.copy %403 : tensor<1x64x4096xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x4096xf32>) -> tensor<1x64x64x64xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{2,1,3,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %407 = mhlo.copy %406 : tensor<1x64x64x64xf32>
      mhlo.return %407 : tensor<1x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x64x64x64xf32>
    %148 = "mhlo.fusion"(%arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %392 = mhlo.copy %391 : tensor<4x4x64x64xf32>
      mhlo.return %392 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %149 = stablehlo.convolution(%147, %148) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<1x16x16x64xf32>
    %150 = "mhlo.fusion"(%149, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      mhlo.return %397 : tensor<1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<1x256x64xf32>
    %151 = "mhlo.fusion"(%150, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
      mhlo.return %391 : tensor<1x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
    %152 = "mhlo.fusion"(%151, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x2xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
    %153 = "mhlo.fusion"(%152, %149, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x64xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x64xf32>
      mhlo.return %403 : tensor<1x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<1x256x64xf32>
    %154 = "mhlo.fusion"(%153, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
      mhlo.return %391 : tensor<1x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x64xf32>, tensor<f32>) -> tensor<1x256x2xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x2xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x2xf32>) -> tensor<1x256xf32>
    %156 = "mhlo.fusion"(%arg85, %arg86, %155, %152, %149, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x64xf32>, %arg213: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<1x16x16x64xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x64xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,16,16]{1,3,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x64x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x16x16xf32>) -> tensor<1x64x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,64]{1,2,0}"} : (tensor<1x64x256xf32>) -> tensor<1x256x64xf32>
      %397 = mhlo.copy %396 : tensor<1x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x64xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x64xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x64xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x64xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x256x64xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x64xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x64xf32>) -> tensor<256x64xf32>
      mhlo.return %410 : tensor<256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x64xf32>, tensor<64xf32>) -> tensor<256x64xf32>
    %157 = stablehlo.dot_general %156, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %158 = "mhlo.fusion"(%157, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x64xf32>) -> tensor<1x256x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,32,256]{2,1,3,0}"} : (tensor<1x256x2x32xf32>) -> tensor<1x2x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x2x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x32x256xf32>) -> tensor<2x32x256xf32>
      mhlo.return %396 : tensor<2x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>, tensor<64xf32>) -> tensor<2x32x256xf32>
    %159 = stablehlo.dot_general %146, %158, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4096x256xf32>
    %160 = "mhlo.fusion"(%159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<2x4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
      mhlo.return %393 : tensor<1x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %161 = "mhlo.fusion"(%160, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096x8xf32>
      mhlo.return %391 : tensor<1x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096x8xf32>
    %162 = "mhlo.fusion"(%161, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x2x4096x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4096x8xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
      mhlo.return %391 : tensor<1x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x8xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %163 = "mhlo.fusion"(%162, %159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096xf32>, %arg209: tensor<2x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x4096x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<2x4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x2x4096xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x2x4096xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x2x4096xf32>) -> tensor<2x4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<2x4096xf32>) -> tensor<1x2x4096x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x2x4096x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x2x4096x256xf32>
      mhlo.return %399 : tensor<1x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096xf32>, tensor<2x4096x256xf32>) -> tensor<1x2x4096x256xf32>
    %164 = "mhlo.fusion"(%163, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
      mhlo.return %391 : tensor<1x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<f32>) -> tensor<1x2x4096xf32>
    %165 = "mhlo.fusion"(%163, %164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x2x4096x256xf32>, %arg209: tensor<1x2x4096xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x2x4096xf32>) -> tensor<2x4096xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<2x4096xf32>) -> tensor<1x2x4096x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x2x4096x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x4096x256xf32>) -> tensor<2x4096x256xf32>
      mhlo.return %394 : tensor<2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x4096x256xf32>, tensor<1x2x4096xf32>) -> tensor<2x4096x256xf32>
    %166 = stablehlo.dot_general %156, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x64xf32>, tensor<64x64xf32>) -> tensor<256x64xf32>
    %167 = "mhlo.fusion"(%166, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>, %arg209: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<256x64xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x64xf32>) -> tensor<1x256x2x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,2,256,32]{3,1,2,0}"} : (tensor<1x256x2x32xf32>) -> tensor<1x2x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x2x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x2x256x32xf32>) -> tensor<2x256x32xf32>
      mhlo.return %396 : tensor<2x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>, tensor<64xf32>) -> tensor<2x256x32xf32>
    %168 = stablehlo.dot_general %165, %167, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4096x256xf32>, tensor<2x256x32xf32>) -> tensor<2x4096x32xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<2x4096x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2x4096x32xf32>) -> tensor<1x2x4096x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4096,2,32]{3,1,2,0}"} : (tensor<1x2x4096x32xf32>) -> tensor<1x4096x2x32xf32>
      %393 = mhlo.copy %392 : tensor<1x4096x2x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4096x2x32xf32>) -> tensor<4096x64xf32>
      mhlo.return %394 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<2x4096x32xf32>) -> tensor<4096x64xf32>
    %170 = stablehlo.dot_general %169, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %171 = "mhlo.fusion"(%139, %170, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x64xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x4096x64xf32>
      mhlo.return %394 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %172 = "mhlo.fusion"(%171, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %173 = "mhlo.fusion"(%171, %172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x4096x64xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x4096x64xf32>
      mhlo.return %396 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x4096x64xf32>
    %174 = "mhlo.fusion"(%173, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %176 = "mhlo.fusion"(%arg93, %arg94, %175, %171, %172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %403 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<4096x64xf32>
    %177 = stablehlo.dot_general %176, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %178 = "mhlo.fusion"(%177, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<4096x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<1x4096x256xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,4096]{1,2,0}"} : (tensor<1x4096x256xf32>) -> tensor<1x256x4096xf32>
      %395 = mhlo.copy %394 : tensor<1x256x4096xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4096xf32>) -> tensor<1x256x64x64xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,256]{2,1,3,0}"} : (tensor<1x256x64x64xf32>) -> tensor<1x64x64x256xf32>
      %398 = mhlo.copy %397 : tensor<1x64x64x256xf32>
      mhlo.return %398 : tensor<1x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<1x64x64x256xf32>
    %179 = "mhlo.fusion"(%arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x256xf32>
      mhlo.return %392 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %180 = stablehlo.convolution(%178, %179) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<1x64x64x256xf32>
    %181 = "mhlo.fusion"(%180, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x64x256xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x64x64x256xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x64x64x256xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %396 = mhlo.copy %395 : tensor<1x256x64x64xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,256]{1,2,0}"} : (tensor<1x256x4096xf32>) -> tensor<1x4096x256xf32>
      %399 = mhlo.copy %398 : tensor<1x4096x256xf32>
      %400 = stablehlo.negate %393 : tensor<1x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x64x256xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x64x64x256xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %404 = mhlo.copy %403 : tensor<1x256x64x64xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x64x64xf32>) -> tensor<1x256x4096xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,256]{1,2,0}"} : (tensor<1x256x4096xf32>) -> tensor<1x4096x256xf32>
      %407 = mhlo.copy %406 : tensor<1x4096x256xf32>
      %408 = stablehlo.abs %407 : tensor<1x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x4096x256xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x4096x256xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x4096x256xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x4096x256xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x4096x256xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x4096x256xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x4096x256xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x4096x256xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %436 = stablehlo.negate %412 : tensor<1x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %439 = stablehlo.exponential %436 : tensor<1x4096x256xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x4096x256xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x4096x256xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x4096x256xf32>, tensor<1x4096x256xf32>) -> tensor<1x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x4096x256xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x4096x256xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x4096x256xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x4096x256xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x4096x256xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x4096x256xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x4096x256xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x4096x256xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x4096x256xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x4096x256xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x4096x256xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x4096x256xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x4096x256xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x4096x256xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x4096x256xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x4096x256xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x4096x256xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x4096x256xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x4096x256xi1>, tensor<1x4096x256xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x4096x256xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %498 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x64x64x256xf32>, tensor<256xf32>) -> tensor<4096x256xf32>
    %182 = stablehlo.dot_general %181, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<64x256xf32>) -> tensor<4096x64xf32>
    %183 = "mhlo.fusion"(%171, %182, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<4096x64xf32>, %arg210: tensor<64xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<4096x64xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<1x4096x64xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x4096x64xf32>
      mhlo.return %394 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<4096x64xf32>, tensor<64xf32>) -> tensor<1x4096x64xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %391 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %185 = "mhlo.fusion"(%183, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x4096x64xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x4096x64xf32>
      mhlo.return %396 : tensor<1x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x4096x64xf32>
    %186 = "mhlo.fusion"(%185, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x64xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
      mhlo.return %391 : tensor<1x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x64xf32>, tensor<f32>) -> tensor<1x4096x2xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x4096x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x4096x2xf32>, tensor<f32>) -> tensor<1x4096xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x4096xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x4096xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x4096xf32>
      mhlo.return %396 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096x2xf32>) -> tensor<1x4096xf32>
    %188 = "mhlo.fusion"(%arg101, %arg102, %187, %183, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<1x64x64x64xf32>
      %404 = stablehlo.transpose %403, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,64]{1,3,2,0}"} : (tensor<1x64x64x64xf32>) -> tensor<1x64x64x64xf32>
      %405 = mhlo.copy %404 : tensor<1x64x64x64xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x64xf32>) -> tensor<1x64x4096xf32>
      %407 = stablehlo.transpose %406, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,4096,64]{1,2,0}"} : (tensor<1x64x4096xf32>) -> tensor<1x4096x64xf32>
      %408 = mhlo.copy %407 : tensor<1x4096x64xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %409 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<4096x64xf32>
    %189 = stablehlo.dot_general %188, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<256x64xf32>) -> tensor<4096x256xf32>
    %190 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<31> : tensor<i32>
      %391 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %393 = stablehlo.iota dim = 0 : tensor<128xi32>
      %394 = stablehlo.convert %393 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %396 = stablehlo.add %394, %395 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %398 = stablehlo.multiply %396, %397 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %400 = stablehlo.add %398, %399 : tensor<128xf32>
      %401 = stablehlo.maximum %392, %400 : tensor<128xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %403 = stablehlo.convert %402 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %404 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %405 = stablehlo.add %403, %404 : tensor<128x1xi32>
      %406 = stablehlo.minimum %391, %405 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %408 = stablehlo.compare LT, %406, %407 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<32> : tensor<i32>
      %409 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %410 = stablehlo.add %406, %409 : tensor<128x1xi32>
      %411 = stablehlo.select %408, %410, %406 : tensor<128x1xi1>, tensor<128x1xi32>
      %412 = mhlo.bitcast %411 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %413 = stablehlo.broadcast_in_dim %412, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %414 = stablehlo.convert %401 : (tensor<128xf32>) -> tensor<128xi32>
      %415 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %417 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %418 = stablehlo.add %414, %417 : tensor<128xi32>
      %419 = stablehlo.select %416, %418, %414 : tensor<128xi1>, tensor<128xi32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %421 = stablehlo.concatenate %413, %420, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %421 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %191 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %392 = stablehlo.iota dim = 0 : tensor<128xi32>
      %393 = stablehlo.convert %392 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %395 = stablehlo.add %393, %394 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %399 = stablehlo.add %397, %398 : tensor<128xf32>
      %400 = stablehlo.maximum %391, %399 : tensor<128xf32>
      %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %402 = stablehlo.convert %401 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %403 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %404 = stablehlo.compare LT, %402, %403 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %405 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %406 = stablehlo.add %402, %405 : tensor<128x1xi32>
      %407 = stablehlo.select %404, %406, %402 : tensor<128x1xi1>, tensor<128x1xi32>
      %408 = mhlo.bitcast %407 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %410 = stablehlo.convert %400 : (tensor<128xf32>) -> tensor<128xi32>
      %411 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %412 = stablehlo.compare LT, %410, %411 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %413 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %414 = stablehlo.add %410, %413 : tensor<128xi32>
      %415 = stablehlo.select %412, %414, %410 : tensor<128xi1>, tensor<128xi32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %417 = stablehlo.concatenate %409, %416, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %417 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %192 = "mhlo.fusion"(%arg101, %arg102, %187, %183, %184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<1x4096xf32>, %arg211: tensor<1x4096x64xf32>, %arg212: tensor<1x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4096xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x4096xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x4096x64xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x4096xf32>) -> tensor<4096xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096x64xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x4096x64xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x4096x64xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<1x4096x64xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x4096x64xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4096x64xf32>) -> tensor<1x64x64x64xf32>
      mhlo.return %403 : tensor<1x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<1x4096xf32>, tensor<1x4096x64xf32>, tensor<1x4096xf32>) -> tensor<1x64x64x64xf32>
    %193 = "mhlo.fusion"(%arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x64x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,160]{1,0,2,3}"} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
      %392 = mhlo.copy %391 : tensor<3x3x64x160xf32>
      mhlo.return %392 : tensor<3x3x64x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
    %194 = stablehlo.convolution(%192, %193) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x64xf32>, tensor<3x3x64x160xf32>) -> tensor<1x32x32x160xf32>
    %195 = "mhlo.fusion"(%194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x32x32x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      mhlo.return %397 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %196 = "mhlo.fusion"(%195, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %197 = "mhlo.fusion"(%196, %194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024xf32>, %arg209: tensor<1x32x32x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x1024xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x1024x160xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x1024x160xf32>
      mhlo.return %403 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %198 = "mhlo.fusion"(%197, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %200 = "mhlo.fusion"(%arg106, %arg107, %199, %196, %194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024xf32>, %arg212: tensor<1x32x32x160xf32>, %arg213: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x1024xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x1024x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x1024x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x1024x160xf32>
      mhlo.return %409 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %201 = "mhlo.fusion"(%200, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %202 = "mhlo.fusion"(%201, %arg106, %arg107, %199, %196, %194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<1x1024xf32>, %arg212: tensor<1x1024xf32>, %arg213: tensor<1x32x32x160xf32>, %arg214: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg213, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %399 = stablehlo.multiply %arg212, %398 : tensor<1x1024xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x1024x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x1024x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x1024x160xf32>
      %410 = stablehlo.multiply %arg208, %398 : tensor<1x1024xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x1024x160xf32>
      %414 = stablehlo.multiply %413, %413 : tensor<1x1024x160xf32>
      mhlo.return %414 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %203 = "mhlo.fusion"(%202, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %204 = "mhlo.fusion"(%203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %205 = "mhlo.fusion"(%arg108, %arg109, %204, %201, %arg106, %arg107, %199, %196, %194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<1x1024xf32>, %arg215: tensor<1x1024xf32>, %arg216: tensor<1x32x32x160xf32>, %arg217: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x1024xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x1024x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x1024x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x1024x160xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x1024xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x1024x160xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x1024x160xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %421 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1024x160xf32>
    %206 = stablehlo.dot_general %205, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %207 = "mhlo.fusion"(%206, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,1024,32]{3,1,2,0}"} : (tensor<1x1024x5x32xf32>) -> tensor<1x5x1024x32xf32>
      %395 = mhlo.copy %394 : tensor<1x5x1024x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x1024x32xf32>) -> tensor<5x1024x32xf32>
      mhlo.return %396 : tensor<5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>) -> tensor<5x1024x32xf32>
    %208 = "mhlo.fusion"(%arg108, %arg109, %204, %201, %arg106, %arg107, %199, %196, %194, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<1x1024xf32>, %arg215: tensor<1x1024xf32>, %arg216: tensor<1x32x32x160xf32>, %arg217: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x32x32x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %394 = mhlo.copy %393 : tensor<1x160x32x32xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %397 = mhlo.copy %396 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x1024xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x1024x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x1024x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x1024x160xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x1024xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x1024x160xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x1024x160xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,160,1024]{1,2,0}"} : (tensor<1x1024x160xf32>) -> tensor<1x160x1024xf32>
      %422 = mhlo.copy %421 : tensor<1x160x1024xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,160]{2,1,3,0}"} : (tensor<1x160x32x32xf32>) -> tensor<1x32x32x160xf32>
      %425 = mhlo.copy %424 : tensor<1x32x32x160xf32>
      mhlo.return %425 : tensor<1x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>) -> tensor<1x32x32x160xf32>
    %209 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %392 = mhlo.copy %391 : tensor<2x2x160x160xf32>
      mhlo.return %392 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %210 = stablehlo.convolution(%208, %209) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<1x16x16x160xf32>
    %211 = "mhlo.fusion"(%210, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      mhlo.return %397 : tensor<1x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<1x256x160xf32>
    %212 = "mhlo.fusion"(%211, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
    %213 = "mhlo.fusion"(%212, %210, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x160xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x160xf32>
      mhlo.return %403 : tensor<1x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<1x256x160xf32>
    %214 = "mhlo.fusion"(%213, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256x5xf32>
      mhlo.return %391 : tensor<1x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256x5xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x5xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x5xf32>) -> tensor<1x256xf32>
    %216 = "mhlo.fusion"(%arg114, %arg115, %215, %212, %210, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x160xf32>, %arg213: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x160xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x160xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x160xf32>) -> tensor<256x160xf32>
      mhlo.return %410 : tensor<256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<256x160xf32>
    %217 = stablehlo.dot_general %216, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %218 = "mhlo.fusion"(%217, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x160xf32>) -> tensor<1x256x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,32,256]{2,1,3,0}"} : (tensor<1x256x5x32xf32>) -> tensor<1x5x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x5x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x32x256xf32>) -> tensor<5x32x256xf32>
      mhlo.return %396 : tensor<5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>, tensor<160xf32>) -> tensor<5x32x256xf32>
    %219 = stablehlo.dot_general %207, %218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<5x1024x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
      mhlo.return %393 : tensor<1x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %221 = "mhlo.fusion"(%220, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024x8xf32>
      mhlo.return %391 : tensor<1x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024x8xf32>
    %222 = "mhlo.fusion"(%221, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x5x1024x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x5x1024x8xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
      mhlo.return %391 : tensor<1x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x8xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %223 = "mhlo.fusion"(%222, %219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024xf32>, %arg209: tensor<5x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<5x1024x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x5x1024xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x5x1024xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x5x1024xf32>) -> tensor<5x1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<5x1024xf32>) -> tensor<1x5x1024x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x5x1024x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x5x1024x256xf32>
      mhlo.return %399 : tensor<1x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024xf32>, tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %224 = "mhlo.fusion"(%223, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
      mhlo.return %391 : tensor<1x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %225 = "mhlo.fusion"(%223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<1x5x1024xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x5x1024xf32>) -> tensor<5x1024xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<5x1024xf32>) -> tensor<1x5x1024x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x5x1024x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x1024x256xf32>) -> tensor<5x1024x256xf32>
      mhlo.return %394 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<1x5x1024xf32>) -> tensor<5x1024x256xf32>
    %226 = stablehlo.dot_general %216, %arg118, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %227 = "mhlo.fusion"(%226, %arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x160xf32>) -> tensor<1x256x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,256,32]{3,1,2,0}"} : (tensor<1x256x5x32xf32>) -> tensor<1x5x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x5x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x256x32xf32>) -> tensor<5x256x32xf32>
      mhlo.return %396 : tensor<5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>, tensor<160xf32>) -> tensor<5x256x32xf32>
    %228 = stablehlo.dot_general %225, %227, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
    %229 = "mhlo.fusion"(%228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<5x1024x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x32xf32>) -> tensor<1x5x1024x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,5,32]{3,1,2,0}"} : (tensor<1x5x1024x32xf32>) -> tensor<1x1024x5x32xf32>
      %393 = mhlo.copy %392 : tensor<1x1024x5x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x5x32xf32>) -> tensor<1024x160xf32>
      mhlo.return %394 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x32xf32>) -> tensor<1024x160xf32>
    %230 = stablehlo.dot_general %229, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %231 = "mhlo.fusion"(%230, %arg106, %arg107, %199, %196, %194, %arg105, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<1x1024xf32>, %arg212: tensor<1x1024xf32>, %arg213: tensor<1x32x32x160xf32>, %arg214: tensor<160xf32>, %arg215: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x160xf32>
      %394 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<1x32x32x160xf32>
      %395 = stablehlo.add %arg213, %394 : tensor<1x32x32x160xf32>
      %396 = stablehlo.transpose %395, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %397 = mhlo.copy %396 : tensor<1x160x32x32xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %399 = stablehlo.transpose %398, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %400 = mhlo.copy %399 : tensor<1x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %402 = stablehlo.multiply %arg212, %401 : tensor<1x1024xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %405 = stablehlo.subtract %400, %404 : tensor<1x1024x160xf32>
      %406 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %407 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %408 = stablehlo.multiply %405, %407 : tensor<1x1024x160xf32>
      %409 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %410 = stablehlo.multiply %408, %409 : tensor<1x1024x160xf32>
      %411 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %412 = stablehlo.add %410, %411 : tensor<1x1024x160xf32>
      %413 = stablehlo.add %393, %412 : tensor<1x1024x160xf32>
      mhlo.return %413 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>, tensor<1x32x32x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %232 = "mhlo.fusion"(%231, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %233 = "mhlo.fusion"(%231, %232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x1024x160xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x1024x160xf32>
      mhlo.return %396 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x1024x160xf32>
    %234 = "mhlo.fusion"(%233, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %236 = "mhlo.fusion"(%arg122, %arg123, %235, %231, %232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %403 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1024x160xf32>
    %237 = stablehlo.dot_general %236, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    %238 = "mhlo.fusion"(%237, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<640xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<1024x640xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<1024x640xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,640,1024]{1,2,0}"} : (tensor<1x1024x640xf32>) -> tensor<1x640x1024xf32>
      %395 = mhlo.copy %394 : tensor<1x640x1024xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x640x1024xf32>) -> tensor<1x640x32x32xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,640]{2,1,3,0}"} : (tensor<1x640x32x32xf32>) -> tensor<1x32x32x640xf32>
      %398 = mhlo.copy %397 : tensor<1x32x32x640xf32>
      mhlo.return %398 : tensor<1x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %239 = "mhlo.fusion"(%arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x640xf32>
      mhlo.return %392 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %240 = stablehlo.convolution(%238, %239) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<1x32x32x640xf32>
    %241 = "mhlo.fusion"(%240, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x32x640xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<1x32x32x640xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x32x32x640xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x32x32x640xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,640,32,32]{1,3,2,0}"} : (tensor<1x32x32x640xf32>) -> tensor<1x640x32x32xf32>
      %396 = mhlo.copy %395 : tensor<1x640x32x32xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,640]{1,2,0}"} : (tensor<1x640x1024xf32>) -> tensor<1x1024x640xf32>
      %399 = mhlo.copy %398 : tensor<1x1024x640xf32>
      %400 = stablehlo.negate %393 : tensor<1x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x32x640xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x32x32x640xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,640,32,32]{1,3,2,0}"} : (tensor<1x32x32x640xf32>) -> tensor<1x640x32x32xf32>
      %404 = mhlo.copy %403 : tensor<1x640x32x32xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,640]{1,2,0}"} : (tensor<1x640x1024xf32>) -> tensor<1x1024x640xf32>
      %407 = mhlo.copy %406 : tensor<1x1024x640xf32>
      %408 = stablehlo.abs %407 : tensor<1x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x1024x640xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x1024x640xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x1024x640xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x1024x640xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x1024x640xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x1024x640xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x1024x640xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x1024x640xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %436 = stablehlo.negate %412 : tensor<1x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %439 = stablehlo.exponential %436 : tensor<1x1024x640xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x1024x640xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x1024x640xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x1024x640xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x1024x640xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x1024x640xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x1024x640xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x1024x640xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x1024x640xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x1024x640xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x1024x640xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x1024x640xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x1024x640xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x1024x640xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x1024x640xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x1024x640xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x1024x640xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x1024x640xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x1024x640xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x1024x640xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x1024x640xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
      mhlo.return %498 : tensor<1024x640xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x32x640xf32>, tensor<640xf32>) -> tensor<1024x640xf32>
    %242 = stablehlo.dot_general %241, %arg128, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    %243 = "mhlo.fusion"(%231, %242, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x160xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x1024x160xf32>
      mhlo.return %394 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %244 = "mhlo.fusion"(%243, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %245 = "mhlo.fusion"(%243, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x1024x160xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x1024x160xf32>
      mhlo.return %396 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x1024x160xf32>
    %246 = "mhlo.fusion"(%245, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %248 = "mhlo.fusion"(%arg130, %arg131, %247, %243, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %403 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1024x160xf32>
    %249 = stablehlo.dot_general %248, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %250 = "mhlo.fusion"(%249, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,1024,32]{3,1,2,0}"} : (tensor<1x1024x5x32xf32>) -> tensor<1x5x1024x32xf32>
      %395 = mhlo.copy %394 : tensor<1x5x1024x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x1024x32xf32>) -> tensor<5x1024x32xf32>
      mhlo.return %396 : tensor<5x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x160xf32>, tensor<160xf32>) -> tensor<5x1024x32xf32>
    %251 = "mhlo.fusion"(%arg130, %arg131, %247, %243, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = stablehlo.transpose %402, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,160,1024]{1,2,0}"} : (tensor<1x1024x160xf32>) -> tensor<1x160x1024xf32>
      %404 = mhlo.copy %403 : tensor<1x160x1024xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x160x1024xf32>) -> tensor<1x160x32x32xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,160]{2,1,3,0}"} : (tensor<1x160x32x32xf32>) -> tensor<1x32x32x160xf32>
      %407 = mhlo.copy %406 : tensor<1x32x32x160xf32>
      mhlo.return %407 : tensor<1x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x32x32x160xf32>
    %252 = "mhlo.fusion"(%arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %392 = mhlo.copy %391 : tensor<2x2x160x160xf32>
      mhlo.return %392 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %253 = stablehlo.convolution(%251, %252) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<1x16x16x160xf32>
    %254 = "mhlo.fusion"(%253, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      mhlo.return %397 : tensor<1x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<1x256x160xf32>
    %255 = "mhlo.fusion"(%254, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256xf32>
    %256 = "mhlo.fusion"(%255, %253, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x160xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x160xf32>
      mhlo.return %403 : tensor<1x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<1x256x160xf32>
    %257 = "mhlo.fusion"(%256, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256x5xf32>
      mhlo.return %391 : tensor<1x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x160xf32>, tensor<f32>) -> tensor<1x256x5xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x5xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x5xf32>) -> tensor<1x256xf32>
    %259 = "mhlo.fusion"(%arg136, %arg137, %258, %255, %253, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x160xf32>, %arg213: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<1x16x16x160xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x160xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,16,16]{1,3,2,0}"} : (tensor<1x16x16x160xf32>) -> tensor<1x160x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x160x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x16x16xf32>) -> tensor<1x160x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,160]{1,2,0}"} : (tensor<1x160x256xf32>) -> tensor<1x256x160xf32>
      %397 = mhlo.copy %396 : tensor<1x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x160xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x160xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x160xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x160xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x256x160xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x160xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x160xf32>) -> tensor<256x160xf32>
      mhlo.return %410 : tensor<256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x160xf32>, tensor<160xf32>) -> tensor<256x160xf32>
    %260 = stablehlo.dot_general %259, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %261 = "mhlo.fusion"(%260, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x160xf32>) -> tensor<1x256x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,32,256]{2,1,3,0}"} : (tensor<1x256x5x32xf32>) -> tensor<1x5x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x5x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x32x256xf32>) -> tensor<5x32x256xf32>
      mhlo.return %396 : tensor<5x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>, tensor<160xf32>) -> tensor<5x32x256xf32>
    %262 = stablehlo.dot_general %250, %261, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x32xf32>, tensor<5x32x256xf32>) -> tensor<5x1024x256xf32>
    %263 = "mhlo.fusion"(%262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<5x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<5x1024x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
      mhlo.return %393 : tensor<1x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %264 = "mhlo.fusion"(%263, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024x8xf32>
      mhlo.return %391 : tensor<1x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024x8xf32>
    %265 = "mhlo.fusion"(%264, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x5x1024x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x5x1024x8xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
      mhlo.return %391 : tensor<1x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x8xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %266 = "mhlo.fusion"(%265, %262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024xf32>, %arg209: tensor<5x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<5x1024x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<5x1024x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x5x1024xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x5x1024xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x5x1024xf32>) -> tensor<5x1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<5x1024xf32>) -> tensor<1x5x1024x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x5x1024x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x5x1024x256xf32>
      mhlo.return %399 : tensor<1x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024xf32>, tensor<5x1024x256xf32>) -> tensor<1x5x1024x256xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
      mhlo.return %391 : tensor<1x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<f32>) -> tensor<1x5x1024xf32>
    %268 = "mhlo.fusion"(%266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x5x1024x256xf32>, %arg209: tensor<1x5x1024xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x5x1024xf32>) -> tensor<5x1024xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<5x1024xf32>) -> tensor<1x5x1024x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x5x1024x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x1024x256xf32>) -> tensor<5x1024x256xf32>
      mhlo.return %394 : tensor<5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x5x1024x256xf32>, tensor<1x5x1024xf32>) -> tensor<5x1024x256xf32>
    %269 = stablehlo.dot_general %259, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x160xf32>, tensor<160x160xf32>) -> tensor<256x160xf32>
    %270 = "mhlo.fusion"(%269, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>, %arg209: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<256x160xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x160xf32>) -> tensor<1x256x5x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,5,256,32]{3,1,2,0}"} : (tensor<1x256x5x32xf32>) -> tensor<1x5x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x5x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x5x256x32xf32>) -> tensor<5x256x32xf32>
      mhlo.return %396 : tensor<5x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>, tensor<160xf32>) -> tensor<5x256x32xf32>
    %271 = stablehlo.dot_general %268, %270, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x1024x256xf32>, tensor<5x256x32xf32>) -> tensor<5x1024x32xf32>
    %272 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<5x1024x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<5x1024x32xf32>) -> tensor<1x5x1024x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,5,32]{3,1,2,0}"} : (tensor<1x5x1024x32xf32>) -> tensor<1x1024x5x32xf32>
      %393 = mhlo.copy %392 : tensor<1x1024x5x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x5x32xf32>) -> tensor<1024x160xf32>
      mhlo.return %394 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<5x1024x32xf32>) -> tensor<1024x160xf32>
    %273 = stablehlo.dot_general %272, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<160x160xf32>) -> tensor<1024x160xf32>
    %274 = "mhlo.fusion"(%243, %273, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x160xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x1024x160xf32>
      mhlo.return %394 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %275 = "mhlo.fusion"(%274, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %276 = "mhlo.fusion"(%274, %275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x1024x160xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x1024x160xf32>
      mhlo.return %396 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x1024x160xf32>
    %277 = "mhlo.fusion"(%276, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %278 = "mhlo.fusion"(%277) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %279 = "mhlo.fusion"(%arg144, %arg145, %278, %274, %275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %403 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1024x160xf32>
    %280 = stablehlo.dot_general %279, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<640x160xf32>) -> tensor<1024x640xf32>
    %281 = "mhlo.fusion"(%280, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x640xf32>, %arg209: tensor<640xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<1024x640xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<1024x640xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x640xf32>) -> tensor<1x1024x640xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,640,1024]{1,2,0}"} : (tensor<1x1024x640xf32>) -> tensor<1x640x1024xf32>
      %395 = mhlo.copy %394 : tensor<1x640x1024xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x640x1024xf32>) -> tensor<1x640x32x32xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,640]{2,1,3,0}"} : (tensor<1x640x32x32xf32>) -> tensor<1x32x32x640xf32>
      %398 = mhlo.copy %397 : tensor<1x32x32x640xf32>
      mhlo.return %398 : tensor<1x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
    %282 = "mhlo.fusion"(%arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x640xf32>
      mhlo.return %392 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %283 = stablehlo.convolution(%281, %282) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<1x32x32x640xf32>
    %284 = "mhlo.fusion"(%283, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x32x640xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<1x32x32x640xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x32x32x640xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x32x32x640xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,640,32,32]{1,3,2,0}"} : (tensor<1x32x32x640xf32>) -> tensor<1x640x32x32xf32>
      %396 = mhlo.copy %395 : tensor<1x640x32x32xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,640]{1,2,0}"} : (tensor<1x640x1024xf32>) -> tensor<1x1024x640xf32>
      %399 = mhlo.copy %398 : tensor<1x1024x640xf32>
      %400 = stablehlo.negate %393 : tensor<1x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x32x640xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x32x32x640xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,640,32,32]{1,3,2,0}"} : (tensor<1x32x32x640xf32>) -> tensor<1x640x32x32xf32>
      %404 = mhlo.copy %403 : tensor<1x640x32x32xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x640x32x32xf32>) -> tensor<1x640x1024xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,640]{1,2,0}"} : (tensor<1x640x1024xf32>) -> tensor<1x1024x640xf32>
      %407 = mhlo.copy %406 : tensor<1x1024x640xf32>
      %408 = stablehlo.abs %407 : tensor<1x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x1024x640xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x1024x640xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x1024x640xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x1024x640xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x1024x640xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x1024x640xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x1024x640xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x1024x640xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %436 = stablehlo.negate %412 : tensor<1x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %439 = stablehlo.exponential %436 : tensor<1x1024x640xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x1024x640xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x1024x640xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x1024x640xf32>, tensor<1x1024x640xf32>) -> tensor<1x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x1024x640xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x1024x640xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x1024x640xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x1024x640xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x1024x640xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x1024x640xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x1024x640xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x1024x640xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x1024x640xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x1024x640xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x1024x640xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x1024x640xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x1024x640xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x1024x640xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1024x640xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x1024x640xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x1024x640xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x1024x640xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x1024x640xi1>, tensor<1x1024x640xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x1024x640xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x640xf32>) -> tensor<1024x640xf32>
      mhlo.return %498 : tensor<1024x640xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x32x640xf32>, tensor<640xf32>) -> tensor<1024x640xf32>
    %285 = stablehlo.dot_general %284, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x640xf32>, tensor<160x640xf32>) -> tensor<1024x160xf32>
    %286 = "mhlo.fusion"(%274, %285, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1024x160xf32>, %arg210: tensor<160xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<1024x160xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<1024x160xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x160xf32>) -> tensor<1x1024x160xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x1024x160xf32>
      mhlo.return %394 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1024x160xf32>, tensor<160xf32>) -> tensor<1x1024x160xf32>
    %287 = "mhlo.fusion"(%286, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
      mhlo.return %391 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %288 = "mhlo.fusion"(%286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x1024x160xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x1024x160xf32>
      mhlo.return %396 : tensor<1x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x1024x160xf32>
    %289 = "mhlo.fusion"(%288, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x160xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
      mhlo.return %391 : tensor<1x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x160xf32>, tensor<f32>) -> tensor<1x1024x5xf32>
    %290 = "mhlo.fusion"(%289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x1024x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x5xf32>, tensor<f32>) -> tensor<1x1024xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x1024xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x1024xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x1024xf32>
      mhlo.return %396 : tensor<1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1024x5xf32>) -> tensor<1x1024xf32>
    %291 = "mhlo.fusion"(%arg152, %arg153, %290, %286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1x32x32x160xf32>
      %404 = stablehlo.transpose %403, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,160,32,32]{1,3,2,0}"} : (tensor<1x32x32x160xf32>) -> tensor<1x160x32x32xf32>
      %405 = mhlo.copy %404 : tensor<1x160x32x32xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x160x32x32xf32>) -> tensor<1x160x1024xf32>
      %407 = stablehlo.transpose %406, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,160]{1,2,0}"} : (tensor<1x160x1024xf32>) -> tensor<1x1024x160xf32>
      %408 = mhlo.copy %407 : tensor<1x1024x160xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1024x160xf32>
      mhlo.return %409 : tensor<1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1024x160xf32>
    %292 = stablehlo.dot_general %291, %arg154, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x160xf32>, tensor<256x160xf32>) -> tensor<1024x256xf32>
    %293 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<15> : tensor<i32>
      %391 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %393 = stablehlo.iota dim = 0 : tensor<128xi32>
      %394 = stablehlo.convert %393 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %396 = stablehlo.add %394, %395 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %397 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %398 = stablehlo.multiply %396, %397 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %400 = stablehlo.add %398, %399 : tensor<128xf32>
      %401 = stablehlo.maximum %392, %400 : tensor<128xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %403 = stablehlo.convert %402 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %404 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %405 = stablehlo.add %403, %404 : tensor<128x1xi32>
      %406 = stablehlo.minimum %391, %405 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %407 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %408 = stablehlo.compare LT, %406, %407 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<16> : tensor<i32>
      %409 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %410 = stablehlo.add %406, %409 : tensor<128x1xi32>
      %411 = stablehlo.select %408, %410, %406 : tensor<128x1xi1>, tensor<128x1xi32>
      %412 = mhlo.bitcast %411 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %413 = stablehlo.broadcast_in_dim %412, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %414 = stablehlo.convert %401 : (tensor<128xf32>) -> tensor<128xi32>
      %415 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %416 = stablehlo.compare LT, %414, %415 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %417 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %418 = stablehlo.add %414, %417 : tensor<128xi32>
      %419 = stablehlo.select %416, %418, %414 : tensor<128xi1>, tensor<128xi32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %421 = stablehlo.concatenate %413, %420, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %421 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %294 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %392 = stablehlo.iota dim = 0 : tensor<128xi32>
      %393 = stablehlo.convert %392 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %395 = stablehlo.add %393, %394 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %399 = stablehlo.add %397, %398 : tensor<128xf32>
      %400 = stablehlo.maximum %391, %399 : tensor<128xf32>
      %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %402 = stablehlo.convert %401 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %403 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %404 = stablehlo.compare LT, %402, %403 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<16> : tensor<i32>
      %405 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %406 = stablehlo.add %402, %405 : tensor<128x1xi32>
      %407 = stablehlo.select %404, %406, %402 : tensor<128x1xi1>, tensor<128x1xi32>
      %408 = mhlo.bitcast %407 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %409 = stablehlo.broadcast_in_dim %408, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %410 = stablehlo.convert %400 : (tensor<128xf32>) -> tensor<128xi32>
      %411 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %412 = stablehlo.compare LT, %410, %411 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %413 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %414 = stablehlo.add %410, %413 : tensor<128xi32>
      %415 = stablehlo.select %412, %414, %410 : tensor<128xi1>, tensor<128xi32>
      %416 = stablehlo.broadcast_in_dim %415, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %417 = stablehlo.concatenate %409, %416, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %417 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %295 = "mhlo.fusion"(%arg152, %arg153, %290, %286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<1x1024xf32>, %arg211: tensor<1x1024x160xf32>, %arg212: tensor<1x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1024xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x1024x160xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1024xf32>) -> tensor<1024xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024x160xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x1024x160xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x1024x160xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<1x1024x160xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x1024x160xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x160xf32>) -> tensor<1x32x32x160xf32>
      mhlo.return %403 : tensor<1x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<1x1024xf32>, tensor<1x1024x160xf32>, tensor<1x1024xf32>) -> tensor<1x32x32x160xf32>
    %296 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,160,256]{1,0,2,3}"} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
      %392 = mhlo.copy %391 : tensor<3x3x160x256xf32>
      mhlo.return %392 : tensor<3x3x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
    %297 = stablehlo.convolution(%295, %296) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x160xf32>, tensor<3x3x160x256xf32>) -> tensor<1x16x16x256xf32>
    %298 = "mhlo.fusion"(%297, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x16x16x256xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x256x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %397 = mhlo.copy %396 : tensor<1x256x256xf32>
      mhlo.return %397 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %299 = "mhlo.fusion"(%298, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %300 = "mhlo.fusion"(%299, %297, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<1x16x16x256xf32>, %arg210: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %392 = stablehlo.add %arg209, %391 : tensor<1x16x16x256xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x256x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %397 = mhlo.copy %396 : tensor<1x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x256xf32>
      %403 = stablehlo.multiply %402, %402 : tensor<1x256x256xf32>
      mhlo.return %403 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<1x16x16x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %302 = "mhlo.fusion"(%301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %303 = "mhlo.fusion"(%arg157, %arg158, %302, %299, %297, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x16x16x256xf32>, %arg213: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %392 = stablehlo.add %arg212, %391 : tensor<1x16x16x256xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x256x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %397 = mhlo.copy %396 : tensor<1x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x256xf32>
      %406 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x256xf32>
      %408 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x256xf32>
      mhlo.return %409 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %304 = "mhlo.fusion"(%303, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %305 = "mhlo.fusion"(%304, %arg157, %arg158, %302, %299, %297, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x256xf32>, %arg213: tensor<1x16x16x256xf32>, %arg214: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %392 = stablehlo.add %arg213, %391 : tensor<1x16x16x256xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x256x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %397 = mhlo.copy %396 : tensor<1x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg212, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x256xf32>
      %406 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x256xf32>
      %408 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x256xf32>
      %410 = stablehlo.multiply %arg208, %398 : tensor<1x256xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x256x256xf32>
      %414 = stablehlo.multiply %413, %413 : tensor<1x256x256xf32>
      mhlo.return %414 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %306 = "mhlo.fusion"(%305, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %307 = "mhlo.fusion"(%306) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %308 = "mhlo.fusion"(%arg159, %arg160, %307, %304, %arg157, %arg158, %302, %299, %297, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<1x256xf32>, %arg215: tensor<1x256xf32>, %arg216: tensor<1x16x16x256xf32>, %arg217: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %392 = stablehlo.add %arg216, %391 : tensor<1x16x16x256xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %394 = mhlo.copy %393 : tensor<1x256x16x16xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %397 = mhlo.copy %396 : tensor<1x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %399 = stablehlo.multiply %arg215, %398 : tensor<1x256xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.subtract %397, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %arg214 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %405 = stablehlo.multiply %402, %404 : tensor<1x256x256xf32>
      %406 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<1x256x256xf32>
      %408 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %409 = stablehlo.add %407, %408 : tensor<1x256x256xf32>
      %410 = stablehlo.multiply %arg211, %398 : tensor<1x256xf32>
      %411 = mhlo.bitcast %410 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %412 = stablehlo.broadcast_in_dim %411, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %413 = stablehlo.subtract %409, %412 : tensor<1x256x256xf32>
      %414 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<1x256x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<1x256x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<1x256x256xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %421 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x256xf32>, tensor<256xf32>) -> tensor<256x256xf32>
    %309 = stablehlo.dot_general %308, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %310 = "mhlo.fusion"(%309, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,256,32]{3,1,2,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x8x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %396 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x256x32xf32>
    %311 = stablehlo.dot_general %308, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %312 = "mhlo.fusion"(%311, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,32,256]{2,1,3,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x8x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x256xf32>) -> tensor<8x32x256xf32>
      mhlo.return %396 : tensor<8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x32x256xf32>
    %313 = stablehlo.dot_general %310, %312, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %314 = "mhlo.fusion"(%313) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<8x256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
      mhlo.return %393 : tensor<1x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %315 = "mhlo.fusion"(%314, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256x8xf32>
      mhlo.return %391 : tensor<1x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256x8xf32>
    %316 = "mhlo.fusion"(%315, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x8xf32>, tensor<f32>) -> tensor<1x8x256xf32>
      mhlo.return %391 : tensor<1x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x8xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %317 = "mhlo.fusion"(%316, %313) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256xf32>, %arg209: tensor<8x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<8x256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x256xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x8x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x256xf32>) -> tensor<8x256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<8x256xf32>) -> tensor<1x8x256x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x8x256x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x8x256x256xf32>
      mhlo.return %399 : tensor<1x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256xf32>, tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %318 = "mhlo.fusion"(%317, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
      mhlo.return %391 : tensor<1x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %319 = "mhlo.fusion"(%317, %318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<1x8x256xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x256xf32>) -> tensor<8x256xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<8x256xf32>) -> tensor<1x8x256x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x8x256x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x256xf32>) -> tensor<8x256x256xf32>
      mhlo.return %394 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<1x8x256xf32>) -> tensor<8x256x256xf32>
    %320 = stablehlo.dot_general %308, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %321 = "mhlo.fusion"(%320, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,256,32]{3,1,2,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x8x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %396 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x256x32xf32>
    %322 = stablehlo.dot_general %319, %321, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
    %323 = "mhlo.fusion"(%322) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x32xf32>) -> tensor<1x8x256x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,8,32]{3,1,2,0}"} : (tensor<1x8x256x32xf32>) -> tensor<1x256x8x32xf32>
      %393 = mhlo.copy %392 : tensor<1x256x8x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x8x32xf32>) -> tensor<256x256xf32>
      mhlo.return %394 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x32xf32>) -> tensor<256x256xf32>
    %324 = stablehlo.dot_general %323, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %325 = "mhlo.fusion"(%324, %arg157, %arg158, %302, %299, %297, %arg156, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1x256xf32>, %arg212: tensor<1x256xf32>, %arg213: tensor<1x16x16x256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %394 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<1x16x16x256xf32>
      %395 = stablehlo.add %arg213, %394 : tensor<1x16x16x256xf32>
      %396 = stablehlo.transpose %395, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %397 = mhlo.copy %396 : tensor<1x256x16x16xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %399 = stablehlo.transpose %398, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %400 = mhlo.copy %399 : tensor<1x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %402 = stablehlo.multiply %arg212, %401 : tensor<1x256xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %405 = stablehlo.subtract %400, %404 : tensor<1x256x256xf32>
      %406 = mhlo.bitcast %arg211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %407 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %408 = stablehlo.multiply %405, %407 : tensor<1x256x256xf32>
      %409 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %410 = stablehlo.multiply %408, %409 : tensor<1x256x256xf32>
      %411 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %412 = stablehlo.add %410, %411 : tensor<1x256x256xf32>
      %413 = stablehlo.add %393, %412 : tensor<1x256x256xf32>
      mhlo.return %413 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x16x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %326 = "mhlo.fusion"(%325, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %327 = "mhlo.fusion"(%325, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x256x256xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x256x256xf32>
      mhlo.return %396 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<1x256x256xf32>
    %328 = "mhlo.fusion"(%327, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %329 = "mhlo.fusion"(%328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %330 = "mhlo.fusion"(%arg169, %arg170, %329, %325, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256x256xf32>, %arg212: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x256x256xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x256x256xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x256x256xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %403 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<256x256xf32>
    %331 = stablehlo.dot_general %330, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %332 = "mhlo.fusion"(%331, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<1024xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1x256x1024xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,256]{1,2,0}"} : (tensor<1x256x1024xf32>) -> tensor<1x1024x256xf32>
      %395 = mhlo.copy %394 : tensor<1x1024x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x256xf32>) -> tensor<1x1024x16x16xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,1024]{2,1,3,0}"} : (tensor<1x1024x16x16xf32>) -> tensor<1x16x16x1024xf32>
      %398 = mhlo.copy %397 : tensor<1x16x16x1024xf32>
      mhlo.return %398 : tensor<1x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>, tensor<1024xf32>) -> tensor<1x16x16x1024xf32>
    %333 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x1024xf32>
      mhlo.return %392 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %334 = stablehlo.convolution(%332, %333) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<1x16x16x1024xf32>
    %335 = "mhlo.fusion"(%334, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x16x1024xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<1x16x16x1024xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x16x16x1024xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x16x16x1024xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,16,16]{1,3,2,0}"} : (tensor<1x16x16x1024xf32>) -> tensor<1x1024x16x16xf32>
      %396 = mhlo.copy %395 : tensor<1x1024x16x16xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,1024]{1,2,0}"} : (tensor<1x1024x256xf32>) -> tensor<1x256x1024xf32>
      %399 = mhlo.copy %398 : tensor<1x256x1024xf32>
      %400 = stablehlo.negate %393 : tensor<1x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x16x1024xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x16x16x1024xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,16,16]{1,3,2,0}"} : (tensor<1x16x16x1024xf32>) -> tensor<1x1024x16x16xf32>
      %404 = mhlo.copy %403 : tensor<1x1024x16x16xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,1024]{1,2,0}"} : (tensor<1x1024x256xf32>) -> tensor<1x256x1024xf32>
      %407 = mhlo.copy %406 : tensor<1x256x1024xf32>
      %408 = stablehlo.abs %407 : tensor<1x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x256x1024xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x256x1024xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x256x1024xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x256x1024xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x256x1024xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x256x1024xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x256x1024xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x256x1024xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %436 = stablehlo.negate %412 : tensor<1x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %439 = stablehlo.exponential %436 : tensor<1x256x1024xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x256x1024xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x256x1024xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x256x1024xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x256x1024xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x256x1024xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x256x1024xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x256x1024xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x256x1024xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x256x1024xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x256x1024xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x256x1024xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x256x1024xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x256x1024xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x256x1024xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x256x1024xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x256x1024xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x256x1024xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x256x1024xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x256x1024xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x256x1024xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x1024xf32>) -> tensor<256x1024xf32>
      mhlo.return %498 : tensor<256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x1024xf32>, tensor<1024xf32>) -> tensor<256x1024xf32>
    %336 = stablehlo.dot_general %335, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    %337 = "mhlo.fusion"(%325, %336, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x256x256xf32>
      mhlo.return %394 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %339 = "mhlo.fusion"(%337, %338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x256x256xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x256x256xf32>
      mhlo.return %396 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<1x256x256xf32>
    %340 = "mhlo.fusion"(%339, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %341 = "mhlo.fusion"(%340) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %342 = "mhlo.fusion"(%arg177, %arg178, %341, %337, %338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256x256xf32>, %arg212: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x256x256xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x256x256xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x256x256xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %403 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<256x256xf32>
    %343 = stablehlo.dot_general %342, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %344 = "mhlo.fusion"(%343, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,256,32]{3,1,2,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x8x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %396 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x256x32xf32>
    %345 = stablehlo.dot_general %342, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %346 = "mhlo.fusion"(%345, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,32,256]{2,1,3,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x32x256xf32>
      %395 = mhlo.copy %394 : tensor<1x8x32x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x32x256xf32>) -> tensor<8x32x256xf32>
      mhlo.return %396 : tensor<8x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x32x256xf32>
    %347 = stablehlo.dot_general %344, %346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x32xf32>, tensor<8x32x256xf32>) -> tensor<8x256x256xf32>
    %348 = "mhlo.fusion"(%347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %392 = stablehlo.multiply %arg208, %391 : tensor<8x256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
      mhlo.return %393 : tensor<1x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %349 = "mhlo.fusion"(%348, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256x8xf32>
      mhlo.return %391 : tensor<1x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256x8xf32>
    %350 = "mhlo.fusion"(%349, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x8xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x256x8xf32>, tensor<f32>) -> tensor<1x8x256xf32>
      mhlo.return %391 : tensor<1x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x8xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %351 = "mhlo.fusion"(%350, %347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256xf32>, %arg209: tensor<8x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x256x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<8x256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x256xf32>
      %395 = stablehlo.maximum %394, %arg208 : tensor<1x8x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x256xf32>) -> tensor<8x256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1, 2] : (tensor<8x256xf32>) -> tensor<1x8x256x256xf32>
      %398 = stablehlo.subtract %393, %397 : tensor<1x8x256x256xf32>
      %399 = stablehlo.exponential %398 : tensor<1x8x256x256xf32>
      mhlo.return %399 : tensor<1x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256xf32>, tensor<8x256x256xf32>) -> tensor<1x8x256x256xf32>
    %352 = "mhlo.fusion"(%351, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
      mhlo.return %391 : tensor<1x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<f32>) -> tensor<1x8x256xf32>
    %353 = "mhlo.fusion"(%351, %352) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x8x256x256xf32>, %arg209: tensor<1x8x256xf32>):
      %391 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x256xf32>) -> tensor<8x256xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<8x256xf32>) -> tensor<1x8x256x256xf32>
      %393 = stablehlo.divide %arg208, %392 : tensor<1x8x256x256xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x256xf32>) -> tensor<8x256x256xf32>
      mhlo.return %394 : tensor<8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x256x256xf32>, tensor<1x8x256xf32>) -> tensor<8x256x256xf32>
    %354 = stablehlo.dot_general %342, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %355 = "mhlo.fusion"(%354, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>, %arg209: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x8x32xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,256,32]{3,1,2,0}"} : (tensor<1x256x8x32xf32>) -> tensor<1x8x256x32xf32>
      %395 = mhlo.copy %394 : tensor<1x8x256x32xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x256x32xf32>) -> tensor<8x256x32xf32>
      mhlo.return %396 : tensor<8x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>, tensor<256xf32>) -> tensor<8x256x32xf32>
    %356 = stablehlo.dot_general %353, %355, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x256x256xf32>, tensor<8x256x32xf32>) -> tensor<8x256x32xf32>
    %357 = "mhlo.fusion"(%356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<8x256x32xf32>):
      %391 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x256x32xf32>) -> tensor<1x8x256x32xf32>
      %392 = stablehlo.transpose %391, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,8,32]{3,1,2,0}"} : (tensor<1x8x256x32xf32>) -> tensor<1x256x8x32xf32>
      %393 = mhlo.copy %392 : tensor<1x256x8x32xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x8x32xf32>) -> tensor<256x256xf32>
      mhlo.return %394 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x32xf32>) -> tensor<256x256xf32>
    %358 = stablehlo.dot_general %357, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %359 = "mhlo.fusion"(%337, %358, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x256x256xf32>
      mhlo.return %394 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %360 = "mhlo.fusion"(%359, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %361 = "mhlo.fusion"(%359, %360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x256x256xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x256x256xf32>
      mhlo.return %396 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<1x256x256xf32>
    %362 = "mhlo.fusion"(%361, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %363 = "mhlo.fusion"(%362) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %364 = "mhlo.fusion"(%arg187, %arg188, %363, %359, %360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256x256xf32>, %arg212: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x256x256xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x256x256xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x256x256xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %403 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<256x256xf32>
    %365 = stablehlo.dot_general %364, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<1024x256xf32>) -> tensor<256x1024xf32>
    %366 = "mhlo.fusion"(%365, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024xf32>, %arg209: tensor<1024xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024xf32>
      %392 = stablehlo.add %391, %arg208 : tensor<256x1024xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1024xf32>) -> tensor<1x256x1024xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,256]{1,2,0}"} : (tensor<1x256x1024xf32>) -> tensor<1x1024x256xf32>
      %395 = mhlo.copy %394 : tensor<1x1024x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x256xf32>) -> tensor<1x1024x16x16xf32>
      %397 = stablehlo.transpose %396, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,1024]{2,1,3,0}"} : (tensor<1x1024x16x16xf32>) -> tensor<1x16x16x1024xf32>
      %398 = mhlo.copy %397 : tensor<1x16x16x1024xf32>
      mhlo.return %398 : tensor<1x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>, tensor<1024xf32>) -> tensor<1x16x16x1024xf32>
    %367 = "mhlo.fusion"(%arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %392 = mhlo.copy %391 : tensor<3x3x1x1024xf32>
      mhlo.return %392 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %368 = stablehlo.convolution(%366, %367) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<1x16x16x1024xf32>
    %369 = "mhlo.fusion"(%368, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x16x1024xf32>
      %392 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<1x16x16x1024xf32>
      %393 = stablehlo.add %arg208, %392 : tensor<1x16x16x1024xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x16x16x1024xf32>
      %395 = stablehlo.transpose %394, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,16,16]{1,3,2,0}"} : (tensor<1x16x16x1024xf32>) -> tensor<1x1024x16x16xf32>
      %396 = mhlo.copy %395 : tensor<1x1024x16x16xf32>
      %397 = mhlo.bitcast %396 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %398 = stablehlo.transpose %397, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,1024]{1,2,0}"} : (tensor<1x1024x256xf32>) -> tensor<1x256x1024xf32>
      %399 = mhlo.copy %398 : tensor<1x256x1024xf32>
      %400 = stablehlo.negate %393 : tensor<1x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x16x1024xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x16x16x1024xf32>
      %403 = stablehlo.transpose %402, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,1024,16,16]{1,3,2,0}"} : (tensor<1x16x16x1024xf32>) -> tensor<1x1024x16x16xf32>
      %404 = mhlo.copy %403 : tensor<1x1024x16x16xf32>
      %405 = mhlo.bitcast %404 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1024x16x16xf32>) -> tensor<1x1024x256xf32>
      %406 = stablehlo.transpose %405, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,1024]{1,2,0}"} : (tensor<1x1024x256xf32>) -> tensor<1x256x1024xf32>
      %407 = mhlo.copy %406 : tensor<1x256x1024xf32>
      %408 = stablehlo.abs %407 : tensor<1x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %409 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %410 = stablehlo.compare LT, %408, %409 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %412 = stablehlo.multiply %407, %407 : tensor<1x256x1024xf32>
      %413 = stablehlo.multiply %411, %412 : tensor<1x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %415 = stablehlo.add %413, %414 : tensor<1x256x1024xf32>
      %416 = stablehlo.multiply %415, %412 : tensor<1x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %418 = stablehlo.add %416, %417 : tensor<1x256x1024xf32>
      %419 = stablehlo.multiply %418, %412 : tensor<1x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %421 = stablehlo.add %419, %420 : tensor<1x256x1024xf32>
      %422 = stablehlo.multiply %421, %412 : tensor<1x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x256x1024xf32>
      %425 = stablehlo.multiply %424, %412 : tensor<1x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %427 = stablehlo.add %425, %426 : tensor<1x256x1024xf32>
      %428 = stablehlo.multiply %427, %412 : tensor<1x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %429 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %430 = stablehlo.add %428, %429 : tensor<1x256x1024xf32>
      %431 = stablehlo.multiply %407, %430 : tensor<1x256x1024xf32>
      %432 = stablehlo.subtract %409, %431 : tensor<1x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %434 = stablehlo.compare LT, %407, %433 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %436 = stablehlo.negate %412 : tensor<1x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %439 = stablehlo.exponential %436 : tensor<1x256x1024xf32>
      %440 = stablehlo.divide %409, %408 : tensor<1x256x1024xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x256x1024xf32>
      %442 = stablehlo.compare LT, %408, %435 : (tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) -> tensor<1x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %444 = stablehlo.divide %409, %412 : tensor<1x256x1024xf32>
      %445 = stablehlo.multiply %443, %444 : tensor<1x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %447 = stablehlo.add %445, %446 : tensor<1x256x1024xf32>
      %448 = stablehlo.multiply %447, %444 : tensor<1x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<1x256x1024xf32>
      %451 = stablehlo.multiply %450, %444 : tensor<1x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<1x256x1024xf32>
      %454 = stablehlo.multiply %453, %444 : tensor<1x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<1x256x1024xf32>
      %457 = stablehlo.multiply %456, %444 : tensor<1x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x256x1024xf32>
      %460 = stablehlo.multiply %459, %444 : tensor<1x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x256x1024xf32>
      %463 = stablehlo.multiply %462, %444 : tensor<1x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x256x1024xf32>
      %466 = stablehlo.multiply %465, %444 : tensor<1x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %470 = stablehlo.multiply %469, %444 : tensor<1x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<1x256x1024xf32>
      %473 = stablehlo.multiply %472, %444 : tensor<1x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<1x256x1024xf32>
      %476 = stablehlo.multiply %475, %444 : tensor<1x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<1x256x1024xf32>
      %479 = stablehlo.multiply %478, %444 : tensor<1x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x256x1024xf32>
      %482 = stablehlo.multiply %481, %444 : tensor<1x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<1x256x1024xf32>
      %485 = stablehlo.multiply %484, %444 : tensor<1x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %487 = stablehlo.add %485, %486 : tensor<1x256x1024xf32>
      %488 = stablehlo.multiply %487, %444 : tensor<1x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x256x1024xf32>
      %490 = stablehlo.add %488, %489 : tensor<1x256x1024xf32>
      %491 = stablehlo.select %442, %468, %490 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %492 = stablehlo.multiply %441, %491 : tensor<1x256x1024xf32>
      %493 = stablehlo.select %438, %433, %492 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %494 = stablehlo.subtract %435, %493 : tensor<1x256x1024xf32>
      %495 = stablehlo.select %434, %494, %493 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %496 = stablehlo.select %410, %432, %495 : tensor<1x256x1024xi1>, tensor<1x256x1024xf32>
      %497 = stablehlo.multiply %399, %496 : tensor<1x256x1024xf32>
      %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x1024xf32>) -> tensor<256x1024xf32>
      mhlo.return %498 : tensor<256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x1024xf32>, tensor<1024xf32>) -> tensor<256x1024xf32>
    %370 = stablehlo.dot_general %369, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x256xf32>
    %371 = "mhlo.fusion"(%359, %370, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<256x256xf32>, %arg210: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<256x256xf32>
      %392 = stablehlo.add %391, %arg209 : tensor<256x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %394 = stablehlo.add %393, %arg208 : tensor<1x256x256xf32>
      mhlo.return %394 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<256x256xf32>, tensor<256xf32>) -> tensor<1x256x256xf32>
    %372 = "mhlo.fusion"(%371, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
      mhlo.return %391 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256xf32>
    %373 = "mhlo.fusion"(%371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg209, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg208, %394 : tensor<1x256x256xf32>
      %396 = stablehlo.multiply %395, %395 : tensor<1x256x256xf32>
      mhlo.return %396 : tensor<1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<1x256x256xf32>
    %374 = "mhlo.fusion"(%373, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x256xf32>, %arg209: tensor<f32>):
      %391 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %392 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %392 : tensor<f32>
      }) : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
      mhlo.return %391 : tensor<1x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x256xf32>, tensor<f32>) -> tensor<1x256x8xf32>
    %375 = "mhlo.fusion"(%374) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x256x8xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x256x8xf32>, tensor<f32>) -> tensor<1x256xf32>
      %cst_3 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<1x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %395 = stablehlo.add %393, %394 : tensor<1x256xf32>
      %396 = stablehlo.rsqrt %395 : tensor<1x256xf32>
      mhlo.return %396 : tensor<1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8xf32>) -> tensor<1x256xf32>
    %376 = "mhlo.fusion"(%arg195, %arg196, %375, %371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<1x256xf32>, %arg211: tensor<1x256x256xf32>, %arg212: tensor<1x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
      %392 = stablehlo.multiply %arg212, %391 : tensor<1x256xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %395 = stablehlo.subtract %arg211, %394 : tensor<1x256x256xf32>
      %396 = mhlo.bitcast %arg210 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x256xf32>) -> tensor<256xf32>
      %397 = stablehlo.broadcast_in_dim %396, dims = [1] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %398 = stablehlo.multiply %395, %397 : tensor<1x256x256xf32>
      %399 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x256x256xf32>
      %401 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x256x256xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x16x16x256xf32>
      %404 = stablehlo.transpose %403, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,16,16]{1,3,2,0}"} : (tensor<1x16x16x256xf32>) -> tensor<1x256x16x16xf32>
      %405 = mhlo.copy %404 : tensor<1x256x16x16xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x16x16xf32>) -> tensor<1x256x256xf32>
      %407 = stablehlo.transpose %406, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %408 = mhlo.copy %407 : tensor<1x256x256xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<256x256xf32>
      mhlo.return %409 : tensor<256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<1x256xf32>, tensor<1x256x256xf32>, tensor<1x256xf32>) -> tensor<256x256xf32>
    %377 = stablehlo.dot_general %376, %arg197, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %378 = "mhlo.fusion"(%1, %2, %arg198, %83, %84, %85, %86, %arg199, %189, %190, %191, %arg200, %292, %cst, %293, %294, %c, %cst_1, %arg201, %377) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x2xi32>, %arg209: tensor<128x128x2xi32>, %arg210: tensor<256xf32>, %arg211: tensor<16384x256xf32>, %arg212: tensor<128x128x2xi32>, %arg213: tensor<128x128x2xi32>, %arg214: tensor<128xf32>, %arg215: tensor<256xf32>, %arg216: tensor<4096x256xf32>, %arg217: tensor<128x128x2xi32>, %arg218: tensor<128x128x2xi32>, %arg219: tensor<256xf32>, %arg220: tensor<1024x256xf32>, %arg221: tensor<f32>, %arg222: tensor<128x128x2xi32>, %arg223: tensor<128x128x2xi32>, %arg224: tensor<i32>, %arg225: tensor<f32>, %arg226: tensor<256xf32>, %arg227: tensor<256x256xf32>):
      %391 = mhlo.bitcast %arg227 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x256xf32>) -> tensor<1x256x256xf32>
      %392 = stablehlo.broadcast_in_dim %arg226, dims = [2] : (tensor<256xf32>) -> tensor<1x256x256xf32>
      %393 = stablehlo.add %391, %392 : tensor<1x256x256xf32>
      %394 = stablehlo.transpose %393, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,256]{1,2,0}"} : (tensor<1x256x256xf32>) -> tensor<1x256x256xf32>
      %395 = mhlo.copy %394 : tensor<1x256x256xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x256xf32>) -> tensor<1x256x16x16xf32>
      %397 = mhlo.bitcast %arg223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %398 = "stablehlo.gather"(%396, %397) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %399 = mhlo.bitcast %398 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %400 = mhlo.copy %399 : tensor<128x128x1x256xf32>
      %401 = stablehlo.transpose %400, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %402 = mhlo.copy %401 : tensor<1x256x128x128xf32>
      %403 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %405 = stablehlo.multiply %arg214, %404 : tensor<128xf32>
      %406 = stablehlo.broadcast_in_dim %arg225, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %407 = stablehlo.add %405, %406 : tensor<128xf32>
      %408 = stablehlo.maximum %403, %407 : tensor<128xf32>
      %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %410 = stablehlo.convert %409 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %411 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %412 = stablehlo.compare LT, %410, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %414 = stablehlo.add %410, %413 : tensor<128x1xi32>
      %415 = stablehlo.select %412, %414, %410 : tensor<128x1xi1>, tensor<128x1xi32>
      %416 = mhlo.bitcast %415 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %417 = stablehlo.broadcast_in_dim %416, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %418 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %419 = stablehlo.convert %408 : (tensor<128xf32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %420 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %421 = stablehlo.add %419, %420 : tensor<128xi32>
      %422 = stablehlo.minimum %418, %421 : tensor<128xi32>
      %423 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %424 = stablehlo.compare LT, %422, %423 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %425 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %426 = stablehlo.add %422, %425 : tensor<128xi32>
      %427 = stablehlo.select %424, %426, %422 : tensor<128xi1>, tensor<128xi32>
      %428 = stablehlo.broadcast_in_dim %427, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %429 = stablehlo.concatenate %417, %428, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %431 = "stablehlo.gather"(%396, %430) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %433 = mhlo.copy %432 : tensor<128x128x1x256xf32>
      %434 = stablehlo.transpose %433, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %435 = mhlo.copy %434 : tensor<1x256x128x128xf32>
      %436 = stablehlo.subtract %435, %402 : tensor<1x256x128x128xf32>
      %437 = stablehlo.convert %419 : (tensor<128xi32>) -> tensor<128xf32>
      %438 = stablehlo.subtract %408, %437 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %440 = stablehlo.clamp %403, %438, %439 : tensor<128xf32>
      %441 = stablehlo.broadcast_in_dim %440, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %442 = stablehlo.multiply %436, %441 : tensor<1x256x128x128xf32>
      %443 = stablehlo.add %402, %442 : tensor<1x256x128x128xf32>
      %444 = mhlo.bitcast %arg222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %445 = "stablehlo.gather"(%396, %444) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %447 = mhlo.copy %446 : tensor<128x128x1x256xf32>
      %448 = stablehlo.transpose %447, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %449 = mhlo.copy %448 : tensor<1x256x128x128xf32>
      %450 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %451 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %452 = stablehlo.add %410, %451 : tensor<128x1xi32>
      %453 = stablehlo.minimum %450, %452 : tensor<128x1xi32>
      %454 = stablehlo.compare LT, %453, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %455 = stablehlo.add %453, %413 : tensor<128x1xi32>
      %456 = stablehlo.select %454, %455, %453 : tensor<128x1xi1>, tensor<128x1xi32>
      %457 = mhlo.bitcast %456 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %459 = stablehlo.concatenate %458, %428, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %460 = mhlo.bitcast %459 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %461 = "stablehlo.gather"(%396, %460) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %462 = mhlo.bitcast %461 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %463 = mhlo.copy %462 : tensor<128x128x1x256xf32>
      %464 = stablehlo.transpose %463, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %465 = mhlo.copy %464 : tensor<1x256x128x128xf32>
      %466 = stablehlo.subtract %465, %449 : tensor<1x256x128x128xf32>
      %467 = stablehlo.multiply %466, %441 : tensor<1x256x128x128xf32>
      %468 = stablehlo.add %449, %467 : tensor<1x256x128x128xf32>
      %469 = stablehlo.subtract %468, %443 : tensor<1x256x128x128xf32>
      %470 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %471 = stablehlo.convert %410 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %472 = stablehlo.subtract %409, %471 : tensor<128x1xf32>
      %473 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %474 = stablehlo.clamp %470, %472, %473 : tensor<128x1xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %477 = stablehlo.multiply %469, %476 : tensor<1x256x128x128xf32>
      %478 = stablehlo.add %443, %477 : tensor<1x256x128x128xf32>
      %479 = mhlo.bitcast %arg220 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x1024x256xf32>
      %480 = stablehlo.broadcast_in_dim %arg219, dims = [2] : (tensor<256xf32>) -> tensor<1x1024x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<1x1024x256xf32>
      %482 = stablehlo.transpose %481, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,1024]{1,2,0}"} : (tensor<1x1024x256xf32>) -> tensor<1x256x1024xf32>
      %483 = mhlo.copy %482 : tensor<1x256x1024xf32>
      %484 = mhlo.bitcast %483 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x1024xf32>) -> tensor<1x256x32x32xf32>
      %485 = mhlo.bitcast %arg218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %486 = "stablehlo.gather"(%484, %485) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %487 = mhlo.bitcast %486 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %488 = mhlo.copy %487 : tensor<128x128x1x256xf32>
      %489 = stablehlo.transpose %488, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %490 = mhlo.copy %489 : tensor<1x256x128x128xf32>
      %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %492 = stablehlo.multiply %arg214, %491 : tensor<128xf32>
      %493 = stablehlo.add %492, %406 : tensor<128xf32>
      %494 = stablehlo.maximum %403, %493 : tensor<128xf32>
      %495 = mhlo.bitcast %494 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %496 = stablehlo.convert %495 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %497 = stablehlo.compare LT, %496, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<32> : tensor<i32>
      %498 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %499 = stablehlo.add %496, %498 : tensor<128x1xi32>
      %500 = stablehlo.select %497, %499, %496 : tensor<128x1xi1>, tensor<128x1xi32>
      %501 = mhlo.bitcast %500 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %502 = stablehlo.broadcast_in_dim %501, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_9 = stablehlo.constant dense<31> : tensor<i32>
      %503 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %504 = stablehlo.convert %494 : (tensor<128xf32>) -> tensor<128xi32>
      %505 = stablehlo.add %504, %420 : tensor<128xi32>
      %506 = stablehlo.minimum %503, %505 : tensor<128xi32>
      %507 = stablehlo.compare LT, %506, %423 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %508 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %509 = stablehlo.add %506, %508 : tensor<128xi32>
      %510 = stablehlo.select %507, %509, %506 : tensor<128xi1>, tensor<128xi32>
      %511 = stablehlo.broadcast_in_dim %510, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %512 = stablehlo.concatenate %502, %511, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %513 = mhlo.bitcast %512 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %514 = "stablehlo.gather"(%484, %513) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %515 = mhlo.bitcast %514 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %516 = mhlo.copy %515 : tensor<128x128x1x256xf32>
      %517 = stablehlo.transpose %516, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %518 = mhlo.copy %517 : tensor<1x256x128x128xf32>
      %519 = stablehlo.subtract %518, %490 : tensor<1x256x128x128xf32>
      %520 = stablehlo.convert %504 : (tensor<128xi32>) -> tensor<128xf32>
      %521 = stablehlo.subtract %494, %520 : tensor<128xf32>
      %522 = stablehlo.clamp %403, %521, %439 : tensor<128xf32>
      %523 = stablehlo.broadcast_in_dim %522, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %524 = stablehlo.multiply %519, %523 : tensor<1x256x128x128xf32>
      %525 = stablehlo.add %490, %524 : tensor<1x256x128x128xf32>
      %526 = mhlo.bitcast %arg217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %527 = "stablehlo.gather"(%484, %526) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %528 = mhlo.bitcast %527 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %529 = mhlo.copy %528 : tensor<128x128x1x256xf32>
      %530 = stablehlo.transpose %529, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %531 = mhlo.copy %530 : tensor<1x256x128x128xf32>
      %532 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %533 = stablehlo.add %496, %451 : tensor<128x1xi32>
      %534 = stablehlo.minimum %532, %533 : tensor<128x1xi32>
      %535 = stablehlo.compare LT, %534, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %536 = stablehlo.add %534, %498 : tensor<128x1xi32>
      %537 = stablehlo.select %535, %536, %534 : tensor<128x1xi1>, tensor<128x1xi32>
      %538 = mhlo.bitcast %537 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %539 = stablehlo.broadcast_in_dim %538, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %540 = stablehlo.concatenate %539, %511, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %541 = mhlo.bitcast %540 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %542 = "stablehlo.gather"(%484, %541) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %543 = mhlo.bitcast %542 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %544 = mhlo.copy %543 : tensor<128x128x1x256xf32>
      %545 = stablehlo.transpose %544, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %546 = mhlo.copy %545 : tensor<1x256x128x128xf32>
      %547 = stablehlo.subtract %546, %531 : tensor<1x256x128x128xf32>
      %548 = stablehlo.multiply %547, %523 : tensor<1x256x128x128xf32>
      %549 = stablehlo.add %531, %548 : tensor<1x256x128x128xf32>
      %550 = stablehlo.subtract %549, %525 : tensor<1x256x128x128xf32>
      %551 = stablehlo.convert %496 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %552 = stablehlo.subtract %495, %551 : tensor<128x1xf32>
      %553 = stablehlo.clamp %470, %552, %473 : tensor<128x1xf32>
      %554 = mhlo.bitcast %553 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %555 = stablehlo.broadcast_in_dim %554, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %556 = stablehlo.multiply %550, %555 : tensor<1x256x128x128xf32>
      %557 = stablehlo.add %525, %556 : tensor<1x256x128x128xf32>
      %558 = mhlo.bitcast %arg216 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<1x4096x256xf32>
      %559 = stablehlo.broadcast_in_dim %arg215, dims = [2] : (tensor<256xf32>) -> tensor<1x4096x256xf32>
      %560 = stablehlo.add %558, %559 : tensor<1x4096x256xf32>
      %561 = stablehlo.transpose %560, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,4096]{1,2,0}"} : (tensor<1x4096x256xf32>) -> tensor<1x256x4096xf32>
      %562 = mhlo.copy %561 : tensor<1x256x4096xf32>
      %563 = mhlo.bitcast %562 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4096xf32>) -> tensor<1x256x64x64xf32>
      %564 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %565 = "stablehlo.gather"(%563, %564) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %566 = mhlo.bitcast %565 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %567 = mhlo.copy %566 : tensor<128x128x1x256xf32>
      %568 = stablehlo.transpose %567, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %569 = mhlo.copy %568 : tensor<1x256x128x128xf32>
      %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %570 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %571 = stablehlo.multiply %arg214, %570 : tensor<128xf32>
      %572 = stablehlo.add %571, %406 : tensor<128xf32>
      %573 = stablehlo.maximum %403, %572 : tensor<128xf32>
      %574 = mhlo.bitcast %573 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %575 = stablehlo.convert %574 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %576 = stablehlo.compare LT, %575, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<64> : tensor<i32>
      %577 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %578 = stablehlo.add %575, %577 : tensor<128x1xi32>
      %579 = stablehlo.select %576, %578, %575 : tensor<128x1xi1>, tensor<128x1xi32>
      %580 = mhlo.bitcast %579 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %581 = stablehlo.broadcast_in_dim %580, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_12 = stablehlo.constant dense<63> : tensor<i32>
      %582 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %583 = stablehlo.convert %573 : (tensor<128xf32>) -> tensor<128xi32>
      %584 = stablehlo.add %583, %420 : tensor<128xi32>
      %585 = stablehlo.minimum %582, %584 : tensor<128xi32>
      %586 = stablehlo.compare LT, %585, %423 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %587 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %588 = stablehlo.add %585, %587 : tensor<128xi32>
      %589 = stablehlo.select %586, %588, %585 : tensor<128xi1>, tensor<128xi32>
      %590 = stablehlo.broadcast_in_dim %589, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %591 = stablehlo.concatenate %581, %590, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %592 = mhlo.bitcast %591 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %593 = "stablehlo.gather"(%563, %592) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %594 = mhlo.bitcast %593 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %595 = mhlo.copy %594 : tensor<128x128x1x256xf32>
      %596 = stablehlo.transpose %595, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %597 = mhlo.copy %596 : tensor<1x256x128x128xf32>
      %598 = stablehlo.subtract %597, %569 : tensor<1x256x128x128xf32>
      %599 = stablehlo.convert %583 : (tensor<128xi32>) -> tensor<128xf32>
      %600 = stablehlo.subtract %573, %599 : tensor<128xf32>
      %601 = stablehlo.clamp %403, %600, %439 : tensor<128xf32>
      %602 = stablehlo.broadcast_in_dim %601, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %603 = stablehlo.multiply %598, %602 : tensor<1x256x128x128xf32>
      %604 = stablehlo.add %569, %603 : tensor<1x256x128x128xf32>
      %605 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %606 = "stablehlo.gather"(%563, %605) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %607 = mhlo.bitcast %606 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %608 = mhlo.copy %607 : tensor<128x128x1x256xf32>
      %609 = stablehlo.transpose %608, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %610 = mhlo.copy %609 : tensor<1x256x128x128xf32>
      %611 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %612 = stablehlo.add %575, %451 : tensor<128x1xi32>
      %613 = stablehlo.minimum %611, %612 : tensor<128x1xi32>
      %614 = stablehlo.compare LT, %613, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %615 = stablehlo.add %613, %577 : tensor<128x1xi32>
      %616 = stablehlo.select %614, %615, %613 : tensor<128x1xi1>, tensor<128x1xi32>
      %617 = mhlo.bitcast %616 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %618 = stablehlo.broadcast_in_dim %617, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %619 = stablehlo.concatenate %618, %590, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %620 = mhlo.bitcast %619 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %621 = "stablehlo.gather"(%563, %620) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %622 = mhlo.bitcast %621 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %623 = mhlo.copy %622 : tensor<128x128x1x256xf32>
      %624 = stablehlo.transpose %623, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %625 = mhlo.copy %624 : tensor<1x256x128x128xf32>
      %626 = stablehlo.subtract %625, %610 : tensor<1x256x128x128xf32>
      %627 = stablehlo.multiply %626, %602 : tensor<1x256x128x128xf32>
      %628 = stablehlo.add %610, %627 : tensor<1x256x128x128xf32>
      %629 = stablehlo.subtract %628, %604 : tensor<1x256x128x128xf32>
      %630 = stablehlo.convert %575 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %631 = stablehlo.subtract %574, %630 : tensor<128x1xf32>
      %632 = stablehlo.clamp %470, %631, %473 : tensor<128x1xf32>
      %633 = mhlo.bitcast %632 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %634 = stablehlo.broadcast_in_dim %633, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %635 = stablehlo.multiply %629, %634 : tensor<1x256x128x128xf32>
      %636 = stablehlo.add %604, %635 : tensor<1x256x128x128xf32>
      %637 = mhlo.bitcast %arg211 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x16384x256xf32>
      %638 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<1x16384x256xf32>
      %639 = stablehlo.add %637, %638 : tensor<1x16384x256xf32>
      %640 = stablehlo.transpose %639, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,256,16384]{1,2,0}"} : (tensor<1x16384x256xf32>) -> tensor<1x256x16384xf32>
      %641 = mhlo.copy %640 : tensor<1x256x16384xf32>
      %642 = mhlo.bitcast %641 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x16384xf32>) -> tensor<1x256x128x128xf32>
      %643 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %644 = "stablehlo.gather"(%642, %643) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %646 = mhlo.copy %645 : tensor<128x128x1x256xf32>
      %647 = stablehlo.transpose %646, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %648 = mhlo.copy %647 : tensor<1x256x128x128xf32>
      %649 = stablehlo.iota dim = 0 : tensor<128xi32>
      %650 = stablehlo.convert %649 : (tensor<128xi32>) -> tensor<128xf32>
      %651 = stablehlo.maximum %403, %650 : tensor<128xf32>
      %652 = mhlo.bitcast %651 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %653 = stablehlo.convert %652 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %654 = stablehlo.compare LT, %653, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_13 = stablehlo.constant dense<128> : tensor<i32>
      %655 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %656 = stablehlo.add %653, %655 : tensor<128x1xi32>
      %657 = stablehlo.select %654, %656, %653 : tensor<128x1xi1>, tensor<128x1xi32>
      %658 = mhlo.bitcast %657 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %659 = stablehlo.broadcast_in_dim %658, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_14 = stablehlo.constant dense<127> : tensor<i32>
      %660 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %661 = stablehlo.convert %651 : (tensor<128xf32>) -> tensor<128xi32>
      %662 = stablehlo.add %661, %420 : tensor<128xi32>
      %663 = stablehlo.minimum %660, %662 : tensor<128xi32>
      %664 = stablehlo.compare LT, %663, %423 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %665 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %666 = stablehlo.add %663, %665 : tensor<128xi32>
      %667 = stablehlo.select %664, %666, %663 : tensor<128xi1>, tensor<128xi32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %669 = stablehlo.concatenate %659, %668, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %670 = mhlo.bitcast %669 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %671 = "stablehlo.gather"(%642, %670) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %672 = mhlo.bitcast %671 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %673 = mhlo.copy %672 : tensor<128x128x1x256xf32>
      %674 = stablehlo.transpose %673, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %675 = mhlo.copy %674 : tensor<1x256x128x128xf32>
      %676 = stablehlo.subtract %675, %648 : tensor<1x256x128x128xf32>
      %677 = stablehlo.convert %661 : (tensor<128xi32>) -> tensor<128xf32>
      %678 = stablehlo.subtract %651, %677 : tensor<128xf32>
      %679 = stablehlo.clamp %403, %678, %439 : tensor<128xf32>
      %680 = stablehlo.broadcast_in_dim %679, dims = [3] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %681 = stablehlo.multiply %676, %680 : tensor<1x256x128x128xf32>
      %682 = stablehlo.add %648, %681 : tensor<1x256x128x128xf32>
      %683 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %684 = "stablehlo.gather"(%642, %683) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %685 = mhlo.bitcast %684 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %686 = mhlo.copy %685 : tensor<128x128x1x256xf32>
      %687 = stablehlo.transpose %686, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %688 = mhlo.copy %687 : tensor<1x256x128x128xf32>
      %689 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %690 = stablehlo.add %653, %451 : tensor<128x1xi32>
      %691 = stablehlo.minimum %689, %690 : tensor<128x1xi32>
      %692 = stablehlo.compare LT, %691, %411 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %693 = stablehlo.add %691, %655 : tensor<128x1xi32>
      %694 = stablehlo.select %692, %693, %691 : tensor<128x1xi1>, tensor<128x1xi32>
      %695 = mhlo.bitcast %694 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %696 = stablehlo.broadcast_in_dim %695, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %697 = stablehlo.concatenate %696, %668, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %698 = mhlo.bitcast %697 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %699 = "stablehlo.gather"(%642, %698) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x1x256x1x1xf32>
      %700 = mhlo.bitcast %699 {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[128,128,1,256]{3,1,0,2}"} : (tensor<16384x1x256x1x1xf32>) -> tensor<128x128x1x256xf32>
      %701 = mhlo.copy %700 : tensor<128x128x1x256xf32>
      %702 = stablehlo.transpose %701, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[1,256,128,128]{1,0,3,2}"} : (tensor<128x128x1x256xf32>) -> tensor<1x256x128x128xf32>
      %703 = mhlo.copy %702 : tensor<1x256x128x128xf32>
      %704 = stablehlo.subtract %703, %688 : tensor<1x256x128x128xf32>
      %705 = stablehlo.multiply %704, %680 : tensor<1x256x128x128xf32>
      %706 = stablehlo.add %688, %705 : tensor<1x256x128x128xf32>
      %707 = stablehlo.subtract %706, %682 : tensor<1x256x128x128xf32>
      %708 = stablehlo.convert %653 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %709 = stablehlo.subtract %652, %708 : tensor<128x1xf32>
      %710 = stablehlo.clamp %470, %709, %473 : tensor<128x1xf32>
      %711 = mhlo.bitcast %710 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %712 = stablehlo.broadcast_in_dim %711, dims = [2] : (tensor<128xf32>) -> tensor<1x256x128x128xf32>
      %713 = stablehlo.multiply %707, %712 : tensor<1x256x128x128xf32>
      %714 = stablehlo.add %682, %713 : tensor<1x256x128x128xf32>
      %715 = stablehlo.concatenate %478, %557, %636, %714, dim = 1 : (tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>, tensor<1x256x128x128xf32>) -> tensor<1x1024x128x128xf32>
      %716 = stablehlo.transpose %715, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,1024]{2,1,3,0}"} : (tensor<1x1024x128x128xf32>) -> tensor<1x128x128x1024xf32>
      %717 = mhlo.copy %716 : tensor<1x128x128x1024xf32>
      mhlo.return %717 : tensor<1x128x128x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<256xf32>, tensor<16384x256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<128xf32>, tensor<256xf32>, tensor<4096x256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<256xf32>, tensor<1024x256xf32>, tensor<f32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<i32>, tensor<f32>, tensor<256xf32>, tensor<256x256xf32>) -> tensor<1x128x128x1024xf32>
    %379 = mhlo.bitcast %378 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x1024xf32>) -> tensor<16384x1024xf32>
    %380 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x1x1xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %392 = mhlo.copy %391 : tensor<1x1x1024x256xf32>
      mhlo.return %392 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %381 = mhlo.bitcast %380 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %382 = stablehlo.dot %379, %381, precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<1024x256xf32>) -> tensor<16384x256xf32>
    %383 = mhlo.bitcast %382 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<1x128x128x256xf32>
    %384 = "mhlo.fusion"(%arg203, %arg204, %0, %383, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<1x128x128x256xf32>, %arg212: tensor<256xf32>):
      %391 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<256xf32>) -> tensor<1x128x128x256xf32>
      %392 = stablehlo.subtract %arg211, %391 : tensor<1x128x128x256xf32>
      %393 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<1x128x128x256xf32>
      %394 = stablehlo.multiply %392, %393 : tensor<1x128x128x256xf32>
      %395 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<1x128x128x256xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<1x128x128x256xf32>
      %397 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<256xf32>) -> tensor<1x128x128x256xf32>
      %398 = stablehlo.add %396, %397 : tensor<1x128x128x256xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x128x128x256xf32>
      %400 = stablehlo.maximum %398, %399 : tensor<1x128x128x256xf32>
      mhlo.return %400 : tensor<1x128x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x128x128x256xf32>, tensor<256xf32>) -> tensor<1x128x128x256xf32>
    %385 = mhlo.bitcast %384 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x256xf32>) -> tensor<16384x256xf32>
    %386 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<150x256x1x1xf32>):
      %391 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,150]{1,0,2,3}"} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
      %392 = mhlo.copy %391 : tensor<1x1x256x150xf32>
      mhlo.return %392 : tensor<1x1x256x150xf32>
    }) {output_operand_aliasing = []} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
    %387 = mhlo.bitcast %386 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x150xf32>) -> tensor<256x150xf32>
    %388 = stablehlo.dot %385, %387, precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x150xf32>) -> tensor<16384x150xf32>
    %389 = mhlo.bitcast %388 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x150xf32>) -> tensor<1x128x128x150xf32>
    %390 = "mhlo.fusion"(%389, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1x128x128x150xf32>, %arg209: tensor<150xf32>):
      %391 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<150xf32>) -> tensor<1x128x128x150xf32>
      %392 = stablehlo.add %arg208, %391 : tensor<1x128x128x150xf32>
      %393 = stablehlo.transpose %392, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,150,128,128]{1,3,2,0}"} : (tensor<1x128x128x150xf32>) -> tensor<1x150x128x128xf32>
      %394 = mhlo.copy %393 : tensor<1x150x128x128xf32>
      mhlo.return %394 : tensor<1x150x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x128x150xf32>, tensor<150xf32>) -> tensor<1x150x128x128xf32>
    return %390 : tensor<1x150x128x128xf32>
  }
}
