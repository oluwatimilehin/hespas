module {
  func.func @main(%arg0: tensor<256xf32>, %arg1: tensor<16x3x512x512xf32>, %arg2: tensor<32x3x7x7xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<32xf32>, %arg7: tensor<32xf32>, %arg8: tensor<32x32xf32>, %arg9: tensor<32xf32>, %arg10: tensor<32x32x8x8xf32>, %arg11: tensor<32xf32>, %arg12: tensor<32xf32>, %arg13: tensor<32xf32>, %arg14: tensor<32x32xf32>, %arg15: tensor<32xf32>, %arg16: tensor<32x32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<128x32xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128x1x3x3xf32>, %arg25: tensor<128xf32>, %arg26: tensor<32x128xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32x32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<32x32x8x8xf32>, %arg33: tensor<32xf32>, %arg34: tensor<32xf32>, %arg35: tensor<32xf32>, %arg36: tensor<32x32xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32x32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<128x32xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x1x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<32x128xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<256x32xf32>, %arg53: tensor<64x32x3x3xf32>, %arg54: tensor<64xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<64xf32>, %arg58: tensor<64xf32>, %arg59: tensor<64x64xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64x64x4x4xf32>, %arg62: tensor<64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<64x64xf32>, %arg66: tensor<64xf32>, %arg67: tensor<64x64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<64x64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64xf32>, %arg73: tensor<256x64xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256x1x3x3xf32>, %arg76: tensor<256xf32>, %arg77: tensor<64x256xf32>, %arg78: tensor<64xf32>, %arg79: tensor<64xf32>, %arg80: tensor<64xf32>, %arg81: tensor<64x64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64x64x4x4xf32>, %arg84: tensor<64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64xf32>, %arg87: tensor<64x64xf32>, %arg88: tensor<64xf32>, %arg89: tensor<64x64xf32>, %arg90: tensor<64xf32>, %arg91: tensor<64x64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64xf32>, %arg95: tensor<256x64xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256x1x3x3xf32>, %arg98: tensor<256xf32>, %arg99: tensor<64x256xf32>, %arg100: tensor<64xf32>, %arg101: tensor<64xf32>, %arg102: tensor<64xf32>, %arg103: tensor<256x64xf32>, %arg104: tensor<160x64x3x3xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160x160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<160x160xf32>, %arg121: tensor<160xf32>, %arg122: tensor<160xf32>, %arg123: tensor<160xf32>, %arg124: tensor<640x160xf32>, %arg125: tensor<640xf32>, %arg126: tensor<640x1x3x3xf32>, %arg127: tensor<640xf32>, %arg128: tensor<160x640xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160x160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<160x160xf32>, %arg143: tensor<160xf32>, %arg144: tensor<160xf32>, %arg145: tensor<160xf32>, %arg146: tensor<640x160xf32>, %arg147: tensor<640xf32>, %arg148: tensor<640x1x3x3xf32>, %arg149: tensor<640xf32>, %arg150: tensor<160x640xf32>, %arg151: tensor<160xf32>, %arg152: tensor<160xf32>, %arg153: tensor<160xf32>, %arg154: tensor<256x160xf32>, %arg155: tensor<256x160x3x3xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<256xf32>, %arg161: tensor<256x256xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256x256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<256x256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256x256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<1024x256xf32>, %arg172: tensor<1024xf32>, %arg173: tensor<1024x1x3x3xf32>, %arg174: tensor<1024xf32>, %arg175: tensor<256x1024xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256x256xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256x256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<256x256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256x256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<1024x256xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<1024x1x3x3xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<256x1024xf32>, %arg194: tensor<256xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256x256xf32>, %arg198: tensor<256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256x1024x1x1xf32>, %arg203: tensor<256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<150x256x1x1xf32>, %arg207: tensor<150xf32>) -> tensor<16x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<256xf32>
      %415 = stablehlo.rsqrt %414 : tensor<256xf32>
      mhlo.return %415 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<127> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.iota dim = 0 : tensor<128xi32>
      %416 = stablehlo.convert %415 : (tensor<128xi32>) -> tensor<128xf32>
      %417 = stablehlo.maximum %414, %416 : tensor<128xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %419 = stablehlo.convert %418 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_4 = stablehlo.constant dense<1> : tensor<i32>
      %420 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %421 = stablehlo.add %419, %420 : tensor<128x1xi32>
      %422 = stablehlo.minimum %413, %421 : tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %423 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %424 = stablehlo.compare LT, %422, %423 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<128> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %422, %425 : tensor<128x1xi32>
      %427 = stablehlo.select %424, %426, %422 : tensor<128x1xi1>, tensor<128x1xi32>
      %428 = mhlo.bitcast %427 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %430 = stablehlo.convert %417 : (tensor<128xf32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %433 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.add %430, %433 : tensor<128xi32>
      %435 = stablehlo.select %432, %434, %430 : tensor<128xi1>, tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %437 = stablehlo.concatenate %429, %436, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %437 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.iota dim = 0 : tensor<128xi32>
      %415 = stablehlo.convert %414 : (tensor<128xi32>) -> tensor<128xf32>
      %416 = stablehlo.maximum %413, %415 : tensor<128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %418 = stablehlo.convert %417 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_3 = stablehlo.constant dense<0> : tensor<i32>
      %419 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %420 = stablehlo.compare LT, %418, %419 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_4 = stablehlo.constant dense<128> : tensor<i32>
      %421 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %422 = stablehlo.add %418, %421 : tensor<128x1xi32>
      %423 = stablehlo.select %420, %422, %418 : tensor<128x1xi1>, tensor<128x1xi32>
      %424 = mhlo.bitcast %423 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %426 = stablehlo.convert %416 : (tensor<128xf32>) -> tensor<128xi32>
      %427 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %428 = stablehlo.compare LT, %426, %427 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %429 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %430 = stablehlo.add %426, %429 : tensor<128xi32>
      %431 = stablehlo.select %428, %430, %426 : tensor<128xi1>, tensor<128xi32>
      %432 = stablehlo.broadcast_in_dim %431, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %433 = stablehlo.concatenate %425, %432, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %433 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x3x512x512xf32>):
      %413 = stablehlo.transpose %arg208, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,512,512,3]{2,1,3,0}"} : (tensor<16x3x512x512xf32>) -> tensor<16x512x512x3xf32>
      %414 = mhlo.copy %413 : tensor<16x512x512x3xf32>
      mhlo.return %414 : tensor<16x512x512x3xf32>
    }) {output_operand_aliasing = []} : (tensor<16x3x512x512xf32>) -> tensor<16x512x512x3xf32>
    %4 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x3x7x7xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,32]{1,0,2,3}"} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
      %414 = mhlo.copy %413 : tensor<7x7x3x32xf32>
      mhlo.return %414 : tensor<7x7x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
    %5 = stablehlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x512x512x3xf32>, tensor<7x7x3x32xf32>) -> tensor<16x128x128x32xf32>
    %6 = "mhlo.fusion"(%5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x128x128x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      mhlo.return %419 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = "mhlo.fusion"(%6, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %8 = "mhlo.fusion"(%7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384xf32>, %arg209: tensor<16x128x128x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x16384xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x16384x32xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %425 = stablehlo.reduce(%424 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %426 = stablehlo.multiply %425, %420 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %428 = stablehlo.add %426, %427 : tensor<16x16384xf32>
      %429 = stablehlo.rsqrt %428 : tensor<16x16384xf32>
      mhlo.return %429 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x16384xf32>
    %9 = "mhlo.fusion"(%arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384xf32>, %arg212: tensor<16x128x128x32xf32>, %arg213: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x16384xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x16384x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x16384x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x16384x32xf32>
      mhlo.return %429 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %10 = "mhlo.fusion"(%9, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %11 = "mhlo.fusion"(%10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<16x16384xf32>, %arg212: tensor<16x16384xf32>, %arg213: tensor<16x128x128x32xf32>, %arg214: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg213, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.multiply %arg212, %420 : tensor<16x16384xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x16384x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x16384x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x16384x32xf32>
      %430 = stablehlo.multiply %arg208, %420 : tensor<16x16384xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x16384x32xf32>
      %433 = stablehlo.multiply %432, %432 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %434 = stablehlo.reduce(%433 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %435 = stablehlo.multiply %434, %420 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x16384xf32>
      %438 = stablehlo.rsqrt %437 : tensor<16x16384xf32>
      mhlo.return %438 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x16384xf32>
    %12 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<16x16384xf32>, %arg215: tensor<16x16384xf32>, %arg216: tensor<16x128x128x32xf32>, %arg217: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x16384xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x16384x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x16384x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x16384x32xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x16384xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x16384x32xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x16384x32xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x16384x32xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x16384x32xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
      mhlo.return %439 : tensor<262144x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<262144x32xf32>
    %13 = stablehlo.dot_general %12, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<32x32xf32>) -> tensor<262144x32xf32>
    %14 = "mhlo.fusion"(%13, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      mhlo.return %415 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %15 = "mhlo.fusion"(%arg6, %arg7, %11, %10, %arg4, %arg5, %8, %7, %5, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<16x16384xf32>, %arg215: tensor<16x16384xf32>, %arg216: tensor<16x128x128x32xf32>, %arg217: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x128x128x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x32x128x128xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %419 = mhlo.copy %418 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x16384xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x16384x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x16384x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x16384x32xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x16384xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x16384x32xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x16384x32xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x16384x32xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x16384x32xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,32,16384]{1,2,0}"} : (tensor<16x16384x32xf32>) -> tensor<16x32x16384xf32>
      %440 = mhlo.copy %439 : tensor<16x32x16384xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x32x16384xf32>) -> tensor<16x32x128x128xf32>
      %442 = stablehlo.transpose %441, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,32]{2,1,3,0}"} : (tensor<16x32x128x128xf32>) -> tensor<16x128x128x32xf32>
      %443 = mhlo.copy %442 : tensor<16x128x128x32xf32>
      mhlo.return %443 : tensor<16x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>) -> tensor<16x128x128x32xf32>
    %16 = "mhlo.fusion"(%arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %414 = mhlo.copy %413 : tensor<8x8x32x32xf32>
      mhlo.return %414 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %17 = stablehlo.convolution(%15, %16) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<16x16x16x32xf32>
    %18 = "mhlo.fusion"(%17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      mhlo.return %419 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<16x256x32xf32>
    %19 = "mhlo.fusion"(%18, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
    %20 = "mhlo.fusion"(%19, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x32xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %425 = stablehlo.reduce(%424 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
      %426 = stablehlo.multiply %425, %420 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %428 = stablehlo.add %426, %427 : tensor<16x256xf32>
      %429 = stablehlo.rsqrt %428 : tensor<16x256xf32>
      mhlo.return %429 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<16x256xf32>
    %21 = "mhlo.fusion"(%arg12, %arg13, %20, %19, %17, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x32xf32>, %arg213: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x256x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x256x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x32xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x32xf32>) -> tensor<4096x32xf32>
      mhlo.return %430 : tensor<4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<4096x32xf32>
    %22 = stablehlo.dot_general %21, %arg14, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x32xf32>, tensor<32x32xf32>) -> tensor<4096x32xf32>
    %23 = "mhlo.fusion"(%22, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4096x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[16,1,256,32]{3,2,0,1}"} : (tensor<4096x32xf32>) -> tensor<16x1x256x32xf32>
      %416 = mhlo.copy %415 : tensor<16x1x256x32xf32>
      %417 = stablehlo.transpose %416, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[16,1,32,256]{2,3,1,0}"} : (tensor<16x1x256x32xf32>) -> tensor<16x1x32x256xf32>
      %418 = mhlo.copy %417 : tensor<16x1x32x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x32x256xf32>) -> tensor<16x32x256xf32>
      mhlo.return %419 : tensor<16x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x32xf32>, tensor<32xf32>) -> tensor<16x32x256xf32>
    %24 = stablehlo.dot_general %14, %23, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16384x32xf32>, tensor<16x32x256xf32>) -> tensor<16x16384x256xf32>
    %25 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x16384x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16,1,16384,256]{3,2,0,1}"} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
      %416 = mhlo.copy %415 : tensor<16x1x16384x256xf32>
      mhlo.return %416 : tensor<16x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
    %26 = "mhlo.fusion"(%cst, %25) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<16x1x16384x256xf32>):
      %413 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %415 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %415 : tensor<f32>
      }) : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384x8xf32>
      %414 = stablehlo.reduce(%413 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x16384x8xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
      mhlo.return %414 : tensor<16x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<16x1x16384x256xf32>) -> tensor<16x1x16384xf32>
    %27 = "mhlo.fusion"(%26, %24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1x16384xf32>, %arg209: tensor<16x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16,1,16384,256]{3,2,0,1}"} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
      %416 = mhlo.copy %415 : tensor<16x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1x16384xf32>
      %418 = stablehlo.maximum %417, %arg208 : tensor<16x1x16384xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x16384xf32>) -> tensor<16x16384xf32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0, 2] : (tensor<16x16384xf32>) -> tensor<16x1x16384x256xf32>
      %421 = stablehlo.subtract %416, %420 : tensor<16x1x16384x256xf32>
      %422 = stablehlo.exponential %421 : tensor<16x1x16384x256xf32>
      mhlo.return %422 : tensor<16x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384xf32>, tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
    %28 = "mhlo.fusion"(%27, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1x16384x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
      mhlo.return %413 : tensor<16x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
    %29 = "mhlo.fusion"(%27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1x16384x256xf32>, %arg209: tensor<16x1x16384xf32>):
      %413 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x16384xf32>) -> tensor<16x16384xf32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [0, 2] : (tensor<16x16384xf32>) -> tensor<16x1x16384x256xf32>
      %415 = stablehlo.divide %arg208, %414 : tensor<16x1x16384x256xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x16384x256xf32>) -> tensor<16x16384x256xf32>
      mhlo.return %416 : tensor<16x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384x256xf32>, tensor<16x1x16384xf32>) -> tensor<16x16384x256xf32>
    %30 = stablehlo.dot_general %21, %arg16, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x32xf32>, tensor<32x32xf32>) -> tensor<4096x32xf32>
    %31 = "mhlo.fusion"(%30, %arg17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4096x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x32xf32>) -> tensor<16x256x32xf32>
      mhlo.return %415 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x32xf32>, tensor<32xf32>) -> tensor<16x256x32xf32>
    %32 = stablehlo.dot_general %29, %31, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16384x256xf32>, tensor<16x256x32xf32>) -> tensor<16x16384x32xf32>
    %33 = mhlo.bitcast %32 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
    %34 = stablehlo.dot_general %33, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<32x32xf32>) -> tensor<262144x32xf32>
    %35 = "mhlo.fusion"(%34, %arg4, %arg5, %8, %7, %5, %arg3, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<16x16384xf32>, %arg212: tensor<16x16384xf32>, %arg213: tensor<16x128x128x32xf32>, %arg214: tensor<32xf32>, %arg215: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<16x128x128x32xf32>
      %417 = stablehlo.add %arg213, %416 : tensor<16x128x128x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %419 = mhlo.copy %418 : tensor<16x32x128x128xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,32]{1,2,0}"} : (tensor<16x32x16384xf32>) -> tensor<16x16384x32xf32>
      %422 = mhlo.copy %421 : tensor<16x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %424 = stablehlo.multiply %arg212, %423 : tensor<16x16384xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %426 = stablehlo.subtract %422, %425 : tensor<16x16384x32xf32>
      %427 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<16x16384x32xf32>
      %429 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<16x16384x32xf32>
      %431 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %432 = stablehlo.add %430, %431 : tensor<16x16384x32xf32>
      %433 = stablehlo.add %415, %432 : tensor<16x16384x32xf32>
      mhlo.return %433 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384xf32>, tensor<16x128x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %36 = "mhlo.fusion"(%35, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %37 = "mhlo.fusion"(%35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.reduce(%417 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %419 = stablehlo.multiply %418, %413 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.add %419, %420 : tensor<16x16384xf32>
      %422 = stablehlo.rsqrt %421 : tensor<16x16384xf32>
      mhlo.return %422 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x16384xf32>
    %38 = "mhlo.fusion"(%arg20, %arg21, %37, %35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
      mhlo.return %423 : tensor<262144x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<262144x32xf32>
    %39 = stablehlo.dot_general %38, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<128x32xf32>) -> tensor<262144x128xf32>
    %40 = "mhlo.fusion"(%39, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x128xf32>, %arg209: tensor<128xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<262144x128xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<262144x128xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x128xf32>) -> tensor<16x16384x128xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,128,16384]{1,2,0}"} : (tensor<16x16384x128xf32>) -> tensor<16x128x16384xf32>
      %417 = mhlo.copy %416 : tensor<16x128x16384xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x128x16384xf32>) -> tensor<16x128x128x128xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{2,1,3,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %420 = mhlo.copy %419 : tensor<16x128x128x128xf32>
      mhlo.return %420 : tensor<16x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x128xf32>, tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %41 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x128xf32>
      mhlo.return %414 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<16x128x128x128xf32>
    %43 = "mhlo.fusion"(%42, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x128x128x128xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x128x128x128xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %418 = mhlo.copy %417 : tensor<16x128x128x128xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x128xf32>) -> tensor<16x128x16384xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,128]{1,2,0}"} : (tensor<16x128x16384xf32>) -> tensor<16x16384x128xf32>
      %421 = mhlo.copy %420 : tensor<16x16384x128xf32>
      %422 = stablehlo.negate %415 : tensor<16x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x128x128x128xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %426 = mhlo.copy %425 : tensor<16x128x128x128xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x128xf32>) -> tensor<16x128x16384xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,128]{1,2,0}"} : (tensor<16x128x16384xf32>) -> tensor<16x16384x128xf32>
      %429 = mhlo.copy %428 : tensor<16x16384x128xf32>
      %430 = stablehlo.abs %429 : tensor<16x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x16384x128xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x16384x128xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x16384x128xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x16384x128xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x16384x128xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x16384x128xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x16384x128xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x16384x128xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %458 = stablehlo.negate %434 : tensor<16x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %461 = stablehlo.exponential %458 : tensor<16x16384x128xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x16384x128xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x16384x128xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x16384x128xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x16384x128xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x16384x128xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x16384x128xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x16384x128xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x16384x128xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x16384x128xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x16384x128xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x16384x128xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x16384x128xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x16384x128xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x16384x128xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x16384x128xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x16384x128xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x16384x128xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x16384x128xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x16384x128xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x16384x128xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x128xf32>) -> tensor<262144x128xf32>
      mhlo.return %520 : tensor<262144x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x128x128xf32>, tensor<128xf32>) -> tensor<262144x128xf32>
    %44 = stablehlo.dot_general %43, %arg26, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x128xf32>, tensor<32x128xf32>) -> tensor<262144x32xf32>
    %45 = "mhlo.fusion"(%35, %44, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<262144x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x16384x32xf32>
      mhlo.return %416 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<262144x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %46 = "mhlo.fusion"(%45, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %47 = "mhlo.fusion"(%45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.reduce(%417 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %419 = stablehlo.multiply %418, %413 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.add %419, %420 : tensor<16x16384xf32>
      %422 = stablehlo.rsqrt %421 : tensor<16x16384xf32>
      mhlo.return %422 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x16384xf32>
    %48 = "mhlo.fusion"(%arg28, %arg29, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
      mhlo.return %423 : tensor<262144x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<262144x32xf32>
    %49 = stablehlo.dot_general %48, %arg30, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<32x32xf32>) -> tensor<262144x32xf32>
    %50 = "mhlo.fusion"(%49, %arg31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      mhlo.return %415 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %51 = "mhlo.fusion"(%arg28, %arg29, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,32,16384]{1,2,0}"} : (tensor<16x16384x32xf32>) -> tensor<16x32x16384xf32>
      %424 = mhlo.copy %423 : tensor<16x32x16384xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x32x16384xf32>) -> tensor<16x32x128x128xf32>
      %426 = stablehlo.transpose %425, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,32]{2,1,3,0}"} : (tensor<16x32x128x128xf32>) -> tensor<16x128x128x32xf32>
      %427 = mhlo.copy %426 : tensor<16x128x128x32xf32>
      mhlo.return %427 : tensor<16x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x128x128x32xf32>
    %52 = "mhlo.fusion"(%arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %414 = mhlo.copy %413 : tensor<8x8x32x32xf32>
      mhlo.return %414 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %53 = stablehlo.convolution(%51, %52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<16x16x16x32xf32>
    %54 = "mhlo.fusion"(%53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      mhlo.return %419 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<16x256x32xf32>
    %55 = "mhlo.fusion"(%54, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
    %56 = "mhlo.fusion"(%55, %53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x32xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %425 = stablehlo.reduce(%424 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x256x32xf32>, tensor<f32>) -> tensor<16x256xf32>
      %426 = stablehlo.multiply %425, %420 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %428 = stablehlo.add %426, %427 : tensor<16x256xf32>
      %429 = stablehlo.rsqrt %428 : tensor<16x256xf32>
      mhlo.return %429 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<16x256xf32>
    %57 = "mhlo.fusion"(%arg34, %arg35, %56, %55, %53, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x32xf32>, %arg213: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<16x16x16x32xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x32xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,16,16]{1,3,2,0}"} : (tensor<16x16x16x32xf32>) -> tensor<16x32x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x32x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x16x16xf32>) -> tensor<16x32x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,32]{1,2,0}"} : (tensor<16x32x256xf32>) -> tensor<16x256x32xf32>
      %419 = mhlo.copy %418 : tensor<16x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x32xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x32xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x32xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x256x32xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x256x32xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x32xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x32xf32>) -> tensor<4096x32xf32>
      mhlo.return %430 : tensor<4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x32xf32>, tensor<32xf32>) -> tensor<4096x32xf32>
    %58 = stablehlo.dot_general %57, %arg36, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x32xf32>, tensor<32x32xf32>) -> tensor<4096x32xf32>
    %59 = "mhlo.fusion"(%58, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4096x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[16,1,256,32]{3,2,0,1}"} : (tensor<4096x32xf32>) -> tensor<16x1x256x32xf32>
      %416 = mhlo.copy %415 : tensor<16x1x256x32xf32>
      %417 = stablehlo.transpose %416, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[16,1,32,256]{2,3,1,0}"} : (tensor<16x1x256x32xf32>) -> tensor<16x1x32x256xf32>
      %418 = mhlo.copy %417 : tensor<16x1x32x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x32x256xf32>) -> tensor<16x32x256xf32>
      mhlo.return %419 : tensor<16x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x32xf32>, tensor<32xf32>) -> tensor<16x32x256xf32>
    %60 = stablehlo.dot_general %50, %59, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16384x32xf32>, tensor<16x32x256xf32>) -> tensor<16x16384x256xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x16384x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16,1,16384,256]{3,2,0,1}"} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
      %416 = mhlo.copy %415 : tensor<16x1x16384x256xf32>
      mhlo.return %416 : tensor<16x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
    %62 = "mhlo.fusion"(%cst, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<16x1x16384x256xf32>):
      %413 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %415 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %415 : tensor<f32>
      }) : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384x8xf32>
      %414 = stablehlo.reduce(%413 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<16x1x16384x8xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
      mhlo.return %414 : tensor<16x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<16x1x16384x256xf32>) -> tensor<16x1x16384xf32>
    %63 = "mhlo.fusion"(%62, %60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1x16384xf32>, %arg209: tensor<16x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[16,1,16384,256]{3,2,0,1}"} : (tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
      %416 = mhlo.copy %415 : tensor<16x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1x16384xf32>
      %418 = stablehlo.maximum %417, %arg208 : tensor<16x1x16384xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x16384xf32>) -> tensor<16x16384xf32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0, 2] : (tensor<16x16384xf32>) -> tensor<16x1x16384x256xf32>
      %421 = stablehlo.subtract %416, %420 : tensor<16x1x16384x256xf32>
      %422 = stablehlo.exponential %421 : tensor<16x1x16384x256xf32>
      mhlo.return %422 : tensor<16x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384xf32>, tensor<16x16384x256xf32>) -> tensor<16x1x16384x256xf32>
    %64 = "mhlo.fusion"(%63, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1x16384x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
      mhlo.return %413 : tensor<16x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384x256xf32>, tensor<f32>) -> tensor<16x1x16384xf32>
    %65 = "mhlo.fusion"(%63, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1x16384x256xf32>, %arg209: tensor<16x1x16384xf32>):
      %413 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1x16384xf32>) -> tensor<16x16384xf32>
      %414 = stablehlo.broadcast_in_dim %413, dims = [0, 2] : (tensor<16x16384xf32>) -> tensor<16x1x16384x256xf32>
      %415 = stablehlo.divide %arg208, %414 : tensor<16x1x16384x256xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x16384x256xf32>) -> tensor<16x16384x256xf32>
      mhlo.return %416 : tensor<16x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1x16384x256xf32>, tensor<16x1x16384xf32>) -> tensor<16x16384x256xf32>
    %66 = stablehlo.dot_general %57, %arg38, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x32xf32>, tensor<32x32xf32>) -> tensor<4096x32xf32>
    %67 = "mhlo.fusion"(%66, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x32xf32>, %arg209: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<4096x32xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x32xf32>) -> tensor<16x256x32xf32>
      mhlo.return %415 : tensor<16x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x32xf32>, tensor<32xf32>) -> tensor<16x256x32xf32>
    %68 = stablehlo.dot_general %65, %67, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16384x256xf32>, tensor<16x256x32xf32>) -> tensor<16x16384x32xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
    %70 = stablehlo.dot_general %69, %arg40, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<32x32xf32>) -> tensor<262144x32xf32>
    %71 = "mhlo.fusion"(%45, %70, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<262144x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x16384x32xf32>
      mhlo.return %416 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<262144x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %72 = "mhlo.fusion"(%71, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.reduce(%417 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %419 = stablehlo.multiply %418, %413 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.add %419, %420 : tensor<16x16384xf32>
      %422 = stablehlo.rsqrt %421 : tensor<16x16384xf32>
      mhlo.return %422 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x16384xf32>
    %74 = "mhlo.fusion"(%arg42, %arg43, %73, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<262144x32xf32>
      mhlo.return %423 : tensor<262144x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<262144x32xf32>
    %75 = stablehlo.dot_general %74, %arg44, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x32xf32>, tensor<128x32xf32>) -> tensor<262144x128xf32>
    %76 = "mhlo.fusion"(%75, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x128xf32>, %arg209: tensor<128xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<262144x128xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<262144x128xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x128xf32>) -> tensor<16x16384x128xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,128,16384]{1,2,0}"} : (tensor<16x16384x128xf32>) -> tensor<16x128x16384xf32>
      %417 = mhlo.copy %416 : tensor<16x128x16384xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x128x16384xf32>) -> tensor<16x128x128x128xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{2,1,3,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %420 = mhlo.copy %419 : tensor<16x128x128x128xf32>
      mhlo.return %420 : tensor<16x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x128xf32>, tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %77 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x128xf32>
      mhlo.return %414 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %78 = stablehlo.convolution(%76, %77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<16x128x128x128xf32>
    %79 = "mhlo.fusion"(%78, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x128x128x128xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x128x128x128xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %418 = mhlo.copy %417 : tensor<16x128x128x128xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x128xf32>) -> tensor<16x128x16384xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,128]{1,2,0}"} : (tensor<16x128x16384xf32>) -> tensor<16x16384x128xf32>
      %421 = mhlo.copy %420 : tensor<16x16384x128xf32>
      %422 = stablehlo.negate %415 : tensor<16x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x128x128x128xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,128]{1,3,2,0}"} : (tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
      %426 = mhlo.copy %425 : tensor<16x128x128x128xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x128xf32>) -> tensor<16x128x16384xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,16384,128]{1,2,0}"} : (tensor<16x128x16384xf32>) -> tensor<16x16384x128xf32>
      %429 = mhlo.copy %428 : tensor<16x16384x128xf32>
      %430 = stablehlo.abs %429 : tensor<16x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x16384x128xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x16384x128xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x16384x128xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x16384x128xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x16384x128xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x16384x128xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x16384x128xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x16384x128xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %458 = stablehlo.negate %434 : tensor<16x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %461 = stablehlo.exponential %458 : tensor<16x16384x128xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x16384x128xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x16384x128xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x16384x128xf32>, tensor<16x16384x128xf32>) -> tensor<16x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x16384x128xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x16384x128xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x16384x128xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x16384x128xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x16384x128xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x16384x128xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x16384x128xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x16384x128xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x16384x128xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x16384x128xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x16384x128xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x16384x128xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x16384x128xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x16384x128xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x16384x128xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x16384x128xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x16384x128xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x16384x128xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x16384x128xi1>, tensor<16x16384x128xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x16384x128xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x128xf32>) -> tensor<262144x128xf32>
      mhlo.return %520 : tensor<262144x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x128x128xf32>, tensor<128xf32>) -> tensor<262144x128xf32>
    %80 = stablehlo.dot_general %79, %arg48, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x128xf32>, tensor<32x128xf32>) -> tensor<262144x32xf32>
    %81 = "mhlo.fusion"(%71, %80, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<262144x32xf32>, %arg210: tensor<32xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<262144x32xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<262144x32xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x32xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x16384x32xf32>
      mhlo.return %416 : tensor<16x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<262144x32xf32>, tensor<32xf32>) -> tensor<16x16384x32xf32>
    %82 = "mhlo.fusion"(%81, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      mhlo.return %413 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
    %83 = "mhlo.fusion"(%81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16384x32xf32>, %arg209: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.reduce(%417 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<16x16384x32xf32>, tensor<f32>) -> tensor<16x16384xf32>
      %419 = stablehlo.multiply %418, %413 : tensor<16x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %421 = stablehlo.add %419, %420 : tensor<16x16384xf32>
      %422 = stablehlo.rsqrt %421 : tensor<16x16384xf32>
      mhlo.return %422 : tensor<16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x16384xf32>
    %84 = "mhlo.fusion"(%arg50, %arg51, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<16x128x128x32xf32>
      %424 = stablehlo.transpose %423, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,128,128]{1,3,2,0}"} : (tensor<16x128x128x32xf32>) -> tensor<16x32x128x128xf32>
      %425 = mhlo.copy %424 : tensor<16x32x128x128xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x32x128x128xf32>) -> tensor<16x32x16384xf32>
      mhlo.return %426 : tensor<16x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x32x16384xf32>
    %85 = "mhlo.fusion"(%arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>):
      %413 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %414 = mhlo.copy %413 : tensor<32x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<16x32x256xf32>
      mhlo.return %415 : tensor<16x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>) -> tensor<16x32x256xf32>
    %86 = stablehlo.dot_general %84, %85, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x32x16384xf32>, tensor<16x32x256xf32>) -> tensor<16x16384x256xf32>
    %87 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<63> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.iota dim = 0 : tensor<128xi32>
      %416 = stablehlo.convert %415 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.add %416, %417 : tensor<128xf32>
      %419 = stablehlo.multiply %418, %417 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %422 = stablehlo.maximum %414, %421 : tensor<128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %424 = stablehlo.convert %423 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<1> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %424, %425 : tensor<128x1xi32>
      %427 = stablehlo.minimum %413, %426 : tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<0> : tensor<i32>
      %428 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %429 = stablehlo.compare LT, %427, %428 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<64> : tensor<i32>
      %430 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %431 = stablehlo.add %427, %430 : tensor<128x1xi32>
      %432 = stablehlo.select %429, %431, %427 : tensor<128x1xi1>, tensor<128x1xi32>
      %433 = mhlo.bitcast %432 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %434 = stablehlo.broadcast_in_dim %433, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %435 = stablehlo.convert %422 : (tensor<128xf32>) -> tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %438 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %439 = stablehlo.add %435, %438 : tensor<128xi32>
      %440 = stablehlo.select %437, %439, %435 : tensor<128xi1>, tensor<128xi32>
      %441 = stablehlo.broadcast_in_dim %440, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %442 = stablehlo.concatenate %434, %441, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %442 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %88 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.iota dim = 0 : tensor<128xi32>
      %415 = stablehlo.convert %414 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %417 = stablehlo.add %415, %416 : tensor<128xf32>
      %418 = stablehlo.multiply %417, %416 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %420 = stablehlo.add %418, %419 : tensor<128xf32>
      %421 = stablehlo.maximum %413, %420 : tensor<128xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %423 = stablehlo.convert %422 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %424 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<64> : tensor<i32>
      %426 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %427 = stablehlo.add %423, %426 : tensor<128x1xi32>
      %428 = stablehlo.select %425, %427, %423 : tensor<128x1xi1>, tensor<128x1xi32>
      %429 = mhlo.bitcast %428 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %430 = stablehlo.broadcast_in_dim %429, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %431 = stablehlo.convert %421 : (tensor<128xf32>) -> tensor<128xi32>
      %432 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %434 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %435 = stablehlo.add %431, %434 : tensor<128xi32>
      %436 = stablehlo.select %433, %435, %431 : tensor<128xi1>, tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %438 = stablehlo.concatenate %430, %437, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %438 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %89 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %413 = stablehlo.iota dim = 0 : tensor<128xi32>
      %414 = stablehlo.convert %413 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %416 = stablehlo.add %414, %415 : tensor<128xf32>
      mhlo.return %416 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %90 = "mhlo.fusion"(%arg50, %arg51, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<16x16384xf32>, %arg211: tensor<16x16384x32xf32>, %arg212: tensor<16x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16384xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x16384xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x16384x32xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x16384xf32>) -> tensor<16x16384x32xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x16384x32xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<16x16384x32xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x16384x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16384x32xf32>) -> tensor<16x128x128x32xf32>
      mhlo.return %423 : tensor<16x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<16x16384xf32>, tensor<16x16384x32xf32>, tensor<16x16384xf32>) -> tensor<16x128x128x32xf32>
    %91 = "mhlo.fusion"(%arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %414 = mhlo.copy %413 : tensor<3x3x32x64xf32>
      mhlo.return %414 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %92 = stablehlo.convolution(%90, %91) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x128x128x32xf32>, tensor<3x3x32x64xf32>) -> tensor<16x64x64x64xf32>
    %93 = "mhlo.fusion"(%92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x64x64x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      mhlo.return %419 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %94 = "mhlo.fusion"(%93, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %95 = "mhlo.fusion"(%94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>, %arg209: tensor<16x64x64x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x4096x64xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x4096x64xf32>
      mhlo.return %424 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %96 = "mhlo.fusion"(%95, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %97 = "mhlo.fusion"(%96, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %98 = "mhlo.fusion"(%97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %99 = "mhlo.fusion"(%arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096xf32>, %arg212: tensor<16x64x64x64xf32>, %arg213: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x4096x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x4096x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x4096x64xf32>
      mhlo.return %429 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %100 = "mhlo.fusion"(%99, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %101 = "mhlo.fusion"(%100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<16x4096xf32>, %arg212: tensor<16x4096xf32>, %arg213: tensor<16x64x64x64xf32>, %arg214: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg213, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %421 = stablehlo.multiply %arg212, %420 : tensor<16x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x4096x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x4096x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x4096x64xf32>
      %430 = stablehlo.multiply %arg208, %420 : tensor<16x4096xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x4096x64xf32>
      %433 = stablehlo.multiply %432, %432 : tensor<16x4096x64xf32>
      mhlo.return %433 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %102 = "mhlo.fusion"(%101, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %103 = "mhlo.fusion"(%102, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %105 = "mhlo.fusion"(%arg57, %arg58, %104, %100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<16x4096xf32>, %arg215: tensor<16x4096xf32>, %arg216: tensor<16x64x64x64xf32>, %arg217: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x4096x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x4096x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x4096x64xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x4096xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x4096x64xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x4096x64xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x4096x64xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x4096x64xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %439 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<65536x64xf32>
    %106 = stablehlo.dot_general %105, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x64xf32>) -> tensor<65536x64xf32>
    %107 = "mhlo.fusion"(%106, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,4096,32]{3,1,2,0}"} : (tensor<16x4096x2x32xf32>) -> tensor<16x2x4096x32xf32>
      %417 = mhlo.copy %416 : tensor<16x2x4096x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x4096x32xf32>) -> tensor<32x4096x32xf32>
      mhlo.return %418 : tensor<32x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>, tensor<64xf32>) -> tensor<32x4096x32xf32>
    %108 = "mhlo.fusion"(%arg57, %arg58, %104, %100, %arg55, %arg56, %98, %94, %92, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<16x4096xf32>, %arg215: tensor<16x4096xf32>, %arg216: tensor<16x64x64x64xf32>, %arg217: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x64x64x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %416 = mhlo.copy %415 : tensor<16x64x64x64xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %419 = mhlo.copy %418 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x4096x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x4096x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x4096x64xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x4096xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x4096x64xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x4096x64xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x4096x64xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x4096x64xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,4096]{1,2,0}"} : (tensor<16x4096x64xf32>) -> tensor<16x64x4096xf32>
      %440 = mhlo.copy %439 : tensor<16x64x4096xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x64x4096xf32>) -> tensor<16x64x64x64xf32>
      %442 = stablehlo.transpose %441, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{2,1,3,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %443 = mhlo.copy %442 : tensor<16x64x64x64xf32>
      mhlo.return %443 : tensor<16x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>) -> tensor<16x64x64x64xf32>
    %109 = "mhlo.fusion"(%arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %414 = mhlo.copy %413 : tensor<4x4x64x64xf32>
      mhlo.return %414 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %110 = stablehlo.convolution(%108, %109) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<16x16x16x64xf32>
    %111 = "mhlo.fusion"(%110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      mhlo.return %419 : tensor<16x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<16x256x64xf32>
    %112 = "mhlo.fusion"(%111, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256xf32>
    %113 = "mhlo.fusion"(%112, %110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x64xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x64xf32>
      mhlo.return %424 : tensor<16x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<16x256x64xf32>
    %114 = "mhlo.fusion"(%113, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256x2xf32>
      mhlo.return %413 : tensor<16x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256x2xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<16x256x2xf32>, tensor<f32>) -> tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %415 = stablehlo.multiply %413, %414 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %417 = stablehlo.add %415, %416 : tensor<16x256xf32>
      %418 = stablehlo.rsqrt %417 : tensor<16x256xf32>
      mhlo.return %418 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x2xf32>) -> tensor<16x256xf32>
    %116 = "mhlo.fusion"(%arg63, %arg64, %115, %112, %110, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x64xf32>, %arg213: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x256x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x256x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x64xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %430 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<4096x64xf32>
    %117 = stablehlo.dot_general %116, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %118 = "mhlo.fusion"(%117, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<16x256x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,32,256]{2,1,3,0}"} : (tensor<16x256x2x32xf32>) -> tensor<16x2x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x2x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x32x256xf32>) -> tensor<32x32x256xf32>
      mhlo.return %418 : tensor<32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<32x32x256xf32>
    %119 = stablehlo.dot_general %107, %118, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x32xf32>, tensor<32x32x256xf32>) -> tensor<32x4096x256xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x4096x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<32x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
      mhlo.return %415 : tensor<16x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
    %121 = "mhlo.fusion"(%120, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096x8xf32>
      mhlo.return %413 : tensor<16x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096x8xf32>
    %122 = "mhlo.fusion"(%121, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x2x4096x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x2x4096x8xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
      mhlo.return %413 : tensor<16x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x8xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
    %123 = "mhlo.fusion"(%122, %119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096xf32>, %arg209: tensor<32x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x4096x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<32x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x2x4096xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x2x4096xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x2x4096xf32>) -> tensor<16x2x4096x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x2x4096x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x2x4096x256xf32>
      mhlo.return %420 : tensor<16x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096xf32>, tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
    %124 = "mhlo.fusion"(%123, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
      mhlo.return %413 : tensor<16x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
    %125 = "mhlo.fusion"(%123, %124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<16x2x4096xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x2x4096xf32>) -> tensor<16x2x4096x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x2x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x4096x256xf32>) -> tensor<32x4096x256xf32>
      mhlo.return %415 : tensor<32x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<16x2x4096xf32>) -> tensor<32x4096x256xf32>
    %126 = stablehlo.dot_general %116, %arg67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %127 = "mhlo.fusion"(%126, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<16x256x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,256,32]{3,1,2,0}"} : (tensor<16x256x2x32xf32>) -> tensor<16x2x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x2x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x256x32xf32>) -> tensor<32x256x32xf32>
      mhlo.return %418 : tensor<32x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<32x256x32xf32>
    %128 = stablehlo.dot_general %125, %127, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x256xf32>, tensor<32x256x32xf32>) -> tensor<32x4096x32xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x4096x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x32xf32>) -> tensor<16x2x4096x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,4096,2,32]{3,1,2,0}"} : (tensor<16x2x4096x32xf32>) -> tensor<16x4096x2x32xf32>
      %415 = mhlo.copy %414 : tensor<16x4096x2x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4096x2x32xf32>) -> tensor<65536x64xf32>
      mhlo.return %416 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<32x4096x32xf32>) -> tensor<65536x64xf32>
    %130 = stablehlo.dot_general %129, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x64xf32>) -> tensor<65536x64xf32>
    %131 = "mhlo.fusion"(%130, %arg55, %arg56, %98, %94, %92, %arg54, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<16x4096xf32>, %arg212: tensor<16x4096xf32>, %arg213: tensor<16x64x64x64xf32>, %arg214: tensor<64xf32>, %arg215: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<16x64x64x64xf32>
      %417 = stablehlo.add %arg213, %416 : tensor<16x64x64x64xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %419 = mhlo.copy %418 : tensor<16x64x64x64xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,64]{1,2,0}"} : (tensor<16x64x4096xf32>) -> tensor<16x4096x64xf32>
      %422 = mhlo.copy %421 : tensor<16x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %424 = stablehlo.multiply %arg212, %423 : tensor<16x4096xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %426 = stablehlo.subtract %422, %425 : tensor<16x4096x64xf32>
      %427 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<16x4096x64xf32>
      %429 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<16x4096x64xf32>
      %431 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %432 = stablehlo.add %430, %431 : tensor<16x4096x64xf32>
      %433 = stablehlo.add %415, %432 : tensor<16x4096x64xf32>
      mhlo.return %433 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096xf32>, tensor<16x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %132 = "mhlo.fusion"(%131, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %133 = "mhlo.fusion"(%131, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x4096x64xf32>
      mhlo.return %417 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
    %134 = "mhlo.fusion"(%133, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %135 = "mhlo.fusion"(%134, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %136 = "mhlo.fusion"(%135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %137 = "mhlo.fusion"(%arg71, %arg72, %136, %131, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %423 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<65536x64xf32>
    %138 = stablehlo.dot_general %137, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<256x64xf32>) -> tensor<65536x256xf32>
    %139 = "mhlo.fusion"(%138, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<65536x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<16x4096x256xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,4096]{1,2,0}"} : (tensor<16x4096x256xf32>) -> tensor<16x256x4096xf32>
      %417 = mhlo.copy %416 : tensor<16x256x4096xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x4096xf32>) -> tensor<16x256x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
      %420 = mhlo.copy %419 : tensor<16x64x64x256xf32>
      mhlo.return %420 : tensor<16x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %140 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x256xf32>
      mhlo.return %414 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %141 = stablehlo.convolution(%139, %140) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<16x64x64x256xf32>
    %142 = "mhlo.fusion"(%141, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x64x64x256xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x64x64x256xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x64x64x256xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %418 = mhlo.copy %417 : tensor<16x256x64x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x64x64xf32>) -> tensor<16x256x4096xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,256]{1,2,0}"} : (tensor<16x256x4096xf32>) -> tensor<16x4096x256xf32>
      %421 = mhlo.copy %420 : tensor<16x4096x256xf32>
      %422 = stablehlo.negate %415 : tensor<16x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x64x64x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x64x64x256xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %426 = mhlo.copy %425 : tensor<16x256x64x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x64x64xf32>) -> tensor<16x256x4096xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,256]{1,2,0}"} : (tensor<16x256x4096xf32>) -> tensor<16x4096x256xf32>
      %429 = mhlo.copy %428 : tensor<16x4096x256xf32>
      %430 = stablehlo.abs %429 : tensor<16x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x4096x256xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x4096x256xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x4096x256xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x4096x256xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x4096x256xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x4096x256xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x4096x256xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x4096x256xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %458 = stablehlo.negate %434 : tensor<16x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %461 = stablehlo.exponential %458 : tensor<16x4096x256xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x4096x256xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x4096x256xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x4096x256xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x4096x256xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x4096x256xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x4096x256xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x4096x256xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x4096x256xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x4096x256xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x4096x256xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x4096x256xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x4096x256xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x4096x256xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x4096x256xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x4096x256xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x4096x256xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x4096x256xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x4096x256xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x4096x256xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x4096x256xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %520 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x64x64x256xf32>, tensor<256xf32>) -> tensor<65536x256xf32>
    %143 = stablehlo.dot_general %142, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<64x256xf32>) -> tensor<65536x64xf32>
    %144 = "mhlo.fusion"(%131, %143, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<65536x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x4096x64xf32>
      mhlo.return %416 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<65536x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %145 = "mhlo.fusion"(%144, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %146 = "mhlo.fusion"(%144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x4096x64xf32>
      mhlo.return %417 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
    %147 = "mhlo.fusion"(%146, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %148 = "mhlo.fusion"(%147, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %150 = "mhlo.fusion"(%arg79, %arg80, %149, %144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %423 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<65536x64xf32>
    %151 = stablehlo.dot_general %150, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x64xf32>) -> tensor<65536x64xf32>
    %152 = "mhlo.fusion"(%151, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,4096,32]{3,1,2,0}"} : (tensor<16x4096x2x32xf32>) -> tensor<16x2x4096x32xf32>
      %417 = mhlo.copy %416 : tensor<16x2x4096x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x4096x32xf32>) -> tensor<32x4096x32xf32>
      mhlo.return %418 : tensor<32x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x64xf32>, tensor<64xf32>) -> tensor<32x4096x32xf32>
    %153 = "mhlo.fusion"(%arg79, %arg80, %149, %144, %145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,64,4096]{1,2,0}"} : (tensor<16x4096x64xf32>) -> tensor<16x64x4096xf32>
      %424 = mhlo.copy %423 : tensor<16x64x4096xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x64x4096xf32>) -> tensor<16x64x64x64xf32>
      %426 = stablehlo.transpose %425, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{2,1,3,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %427 = mhlo.copy %426 : tensor<16x64x64x64xf32>
      mhlo.return %427 : tensor<16x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x64x64x64xf32>
    %154 = "mhlo.fusion"(%arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %414 = mhlo.copy %413 : tensor<4x4x64x64xf32>
      mhlo.return %414 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %155 = stablehlo.convolution(%153, %154) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<16x16x16x64xf32>
    %156 = "mhlo.fusion"(%155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      mhlo.return %419 : tensor<16x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<16x256x64xf32>
    %157 = "mhlo.fusion"(%156, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256xf32>
    %158 = "mhlo.fusion"(%157, %155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x64xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x64xf32>
      mhlo.return %424 : tensor<16x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<16x256x64xf32>
    %159 = "mhlo.fusion"(%158, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256x2xf32>
      mhlo.return %413 : tensor<16x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x64xf32>, tensor<f32>) -> tensor<16x256x2xf32>
    %160 = "mhlo.fusion"(%159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x2xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<16x256x2xf32>, tensor<f32>) -> tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %415 = stablehlo.multiply %413, %414 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %417 = stablehlo.add %415, %416 : tensor<16x256xf32>
      %418 = stablehlo.rsqrt %417 : tensor<16x256xf32>
      mhlo.return %418 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x2xf32>) -> tensor<16x256xf32>
    %161 = "mhlo.fusion"(%arg85, %arg86, %160, %157, %155, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x64xf32>, %arg213: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<16x16x16x64xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x64xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,16,16]{1,3,2,0}"} : (tensor<16x16x16x64xf32>) -> tensor<16x64x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x64x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x16x16xf32>) -> tensor<16x64x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,64]{1,2,0}"} : (tensor<16x64x256xf32>) -> tensor<16x256x64xf32>
      %419 = mhlo.copy %418 : tensor<16x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x64xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x64xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x64xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x256x64xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x256x64xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x64xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x64xf32>) -> tensor<4096x64xf32>
      mhlo.return %430 : tensor<4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x64xf32>, tensor<64xf32>) -> tensor<4096x64xf32>
    %162 = stablehlo.dot_general %161, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %163 = "mhlo.fusion"(%162, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<16x256x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,32,256]{2,1,3,0}"} : (tensor<16x256x2x32xf32>) -> tensor<16x2x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x2x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x32x256xf32>) -> tensor<32x32x256xf32>
      mhlo.return %418 : tensor<32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<32x32x256xf32>
    %164 = stablehlo.dot_general %152, %163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x32xf32>, tensor<32x32x256xf32>) -> tensor<32x4096x256xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x4096x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<32x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
      mhlo.return %415 : tensor<16x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
    %166 = "mhlo.fusion"(%165, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096x8xf32>
      mhlo.return %413 : tensor<16x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096x8xf32>
    %167 = "mhlo.fusion"(%166, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x2x4096x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x2x4096x8xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
      mhlo.return %413 : tensor<16x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x8xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
    %168 = "mhlo.fusion"(%167, %164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096xf32>, %arg209: tensor<32x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x4096x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<32x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x2x4096xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x2x4096xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x2x4096xf32>) -> tensor<16x2x4096x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x2x4096x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x2x4096x256xf32>
      mhlo.return %420 : tensor<16x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096xf32>, tensor<32x4096x256xf32>) -> tensor<16x2x4096x256xf32>
    %169 = "mhlo.fusion"(%168, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
      mhlo.return %413 : tensor<16x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<f32>) -> tensor<16x2x4096xf32>
    %170 = "mhlo.fusion"(%168, %169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x2x4096x256xf32>, %arg209: tensor<16x2x4096xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x2x4096xf32>) -> tensor<16x2x4096x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x2x4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x4096x256xf32>) -> tensor<32x4096x256xf32>
      mhlo.return %415 : tensor<32x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x4096x256xf32>, tensor<16x2x4096xf32>) -> tensor<32x4096x256xf32>
    %171 = stablehlo.dot_general %161, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x64xf32>, tensor<64x64xf32>) -> tensor<4096x64xf32>
    %172 = "mhlo.fusion"(%171, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x64xf32>, %arg209: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<4096x64xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x64xf32>) -> tensor<16x256x2x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,2,256,32]{3,1,2,0}"} : (tensor<16x256x2x32xf32>) -> tensor<16x2x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x2x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x2x256x32xf32>) -> tensor<32x256x32xf32>
      mhlo.return %418 : tensor<32x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x64xf32>, tensor<64xf32>) -> tensor<32x256x32xf32>
    %173 = stablehlo.dot_general %170, %172, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x256xf32>, tensor<32x256x32xf32>) -> tensor<32x4096x32xf32>
    %174 = "mhlo.fusion"(%173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x4096x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<32x4096x32xf32>) -> tensor<16x2x4096x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,4096,2,32]{3,1,2,0}"} : (tensor<16x2x4096x32xf32>) -> tensor<16x4096x2x32xf32>
      %415 = mhlo.copy %414 : tensor<16x4096x2x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x4096x2x32xf32>) -> tensor<65536x64xf32>
      mhlo.return %416 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<32x4096x32xf32>) -> tensor<65536x64xf32>
    %175 = stablehlo.dot_general %174, %arg91, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<64x64xf32>) -> tensor<65536x64xf32>
    %176 = "mhlo.fusion"(%144, %175, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<65536x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x4096x64xf32>
      mhlo.return %416 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<65536x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %177 = "mhlo.fusion"(%176, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %178 = "mhlo.fusion"(%176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x4096x64xf32>
      mhlo.return %417 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
    %179 = "mhlo.fusion"(%178, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %180 = "mhlo.fusion"(%179, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %181 = "mhlo.fusion"(%180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %182 = "mhlo.fusion"(%arg93, %arg94, %181, %176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<65536x64xf32>
      mhlo.return %423 : tensor<65536x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<65536x64xf32>
    %183 = stablehlo.dot_general %182, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x64xf32>, tensor<256x64xf32>) -> tensor<65536x256xf32>
    %184 = "mhlo.fusion"(%183, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<65536x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<65536x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x256xf32>) -> tensor<16x4096x256xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,4096]{1,2,0}"} : (tensor<16x4096x256xf32>) -> tensor<16x256x4096xf32>
      %417 = mhlo.copy %416 : tensor<16x256x4096xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x4096xf32>) -> tensor<16x256x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,256]{2,1,3,0}"} : (tensor<16x256x64x64xf32>) -> tensor<16x64x64x256xf32>
      %420 = mhlo.copy %419 : tensor<16x64x64x256xf32>
      mhlo.return %420 : tensor<16x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x256xf32>, tensor<256xf32>) -> tensor<16x64x64x256xf32>
    %185 = "mhlo.fusion"(%arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x256xf32>
      mhlo.return %414 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %186 = stablehlo.convolution(%184, %185) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<16x64x64x256xf32>
    %187 = "mhlo.fusion"(%186, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x64x64x256xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<16x64x64x256xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x64x64x256xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x64x64x256xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %418 = mhlo.copy %417 : tensor<16x256x64x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x64x64xf32>) -> tensor<16x256x4096xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,256]{1,2,0}"} : (tensor<16x256x4096xf32>) -> tensor<16x4096x256xf32>
      %421 = mhlo.copy %420 : tensor<16x4096x256xf32>
      %422 = stablehlo.negate %415 : tensor<16x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x64x64x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x64x64x256xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,64,64]{1,3,2,0}"} : (tensor<16x64x64x256xf32>) -> tensor<16x256x64x64xf32>
      %426 = mhlo.copy %425 : tensor<16x256x64x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x64x64xf32>) -> tensor<16x256x4096xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,4096,256]{1,2,0}"} : (tensor<16x256x4096xf32>) -> tensor<16x4096x256xf32>
      %429 = mhlo.copy %428 : tensor<16x4096x256xf32>
      %430 = stablehlo.abs %429 : tensor<16x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x4096x256xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x4096x256xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x4096x256xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x4096x256xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x4096x256xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x4096x256xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x4096x256xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x4096x256xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %458 = stablehlo.negate %434 : tensor<16x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %461 = stablehlo.exponential %458 : tensor<16x4096x256xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x4096x256xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x4096x256xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x4096x256xf32>, tensor<16x4096x256xf32>) -> tensor<16x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x4096x256xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x4096x256xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x4096x256xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x4096x256xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x4096x256xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x4096x256xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x4096x256xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x4096x256xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x4096x256xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x4096x256xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x4096x256xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x4096x256xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x4096x256xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x4096x256xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x4096x256xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x4096x256xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x4096x256xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x4096x256xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x4096x256xi1>, tensor<16x4096x256xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x4096x256xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x256xf32>) -> tensor<65536x256xf32>
      mhlo.return %520 : tensor<65536x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x64x64x256xf32>, tensor<256xf32>) -> tensor<65536x256xf32>
    %188 = stablehlo.dot_general %187, %arg99, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x256xf32>, tensor<64x256xf32>) -> tensor<65536x64xf32>
    %189 = "mhlo.fusion"(%176, %188, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<65536x64xf32>, %arg210: tensor<64xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<65536x64xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<65536x64xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x64xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x4096x64xf32>
      mhlo.return %416 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<65536x64xf32>, tensor<64xf32>) -> tensor<16x4096x64xf32>
    %190 = "mhlo.fusion"(%189, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %191 = "mhlo.fusion"(%189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x4096x64xf32>
      mhlo.return %417 : tensor<16x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
    %192 = "mhlo.fusion"(%191, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096x64xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
      mhlo.return %413 : tensor<16x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x64xf32>, tensor<f32>) -> tensor<16x4096x2xf32>
    %193 = "mhlo.fusion"(%192, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x4096x2xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
      mhlo.return %413 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096x2xf32>, tensor<f32>) -> tensor<16x4096xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x4096xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x4096xf32>
      mhlo.return %417 : tensor<16x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %195 = "mhlo.fusion"(%arg101, %arg102, %194, %189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<16x64x64x64xf32>
      %424 = stablehlo.transpose %423, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,64,64,64]{1,3,2,0}"} : (tensor<16x64x64x64xf32>) -> tensor<16x64x64x64xf32>
      %425 = mhlo.copy %424 : tensor<16x64x64x64xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x64x64x64xf32>) -> tensor<16x64x4096xf32>
      mhlo.return %426 : tensor<16x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x64x4096xf32>
    %196 = "mhlo.fusion"(%arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %413 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[64,256]{0,1}"} : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %414 = mhlo.copy %413 : tensor<64x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<16x64x256xf32>
      mhlo.return %415 : tensor<16x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<16x64x256xf32>
    %197 = stablehlo.dot_general %195, %196, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x64x4096xf32>, tensor<16x64x256xf32>) -> tensor<16x4096x256xf32>
    %198 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<31> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.iota dim = 0 : tensor<128xi32>
      %416 = stablehlo.convert %415 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.add %416, %417 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.add %420, %421 : tensor<128xf32>
      %423 = stablehlo.maximum %414, %422 : tensor<128xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %425 = stablehlo.convert %424 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %426 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %427 = stablehlo.add %425, %426 : tensor<128x1xi32>
      %428 = stablehlo.minimum %413, %427 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.compare LT, %428, %429 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<32> : tensor<i32>
      %431 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %432 = stablehlo.add %428, %431 : tensor<128x1xi32>
      %433 = stablehlo.select %430, %432, %428 : tensor<128x1xi1>, tensor<128x1xi32>
      %434 = mhlo.bitcast %433 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %436 = stablehlo.convert %423 : (tensor<128xf32>) -> tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %439 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %440 = stablehlo.add %436, %439 : tensor<128xi32>
      %441 = stablehlo.select %438, %440, %436 : tensor<128xi1>, tensor<128xi32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %443 = stablehlo.concatenate %435, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %443 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %199 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.iota dim = 0 : tensor<128xi32>
      %415 = stablehlo.convert %414 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %417 = stablehlo.add %415, %416 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %422 = stablehlo.maximum %413, %421 : tensor<128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %424 = stablehlo.convert %423 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %427 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %428 = stablehlo.add %424, %427 : tensor<128x1xi32>
      %429 = stablehlo.select %426, %428, %424 : tensor<128x1xi1>, tensor<128x1xi32>
      %430 = mhlo.bitcast %429 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %432 = stablehlo.convert %422 : (tensor<128xf32>) -> tensor<128xi32>
      %433 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.compare LT, %432, %433 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %435 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %436 = stablehlo.add %432, %435 : tensor<128xi32>
      %437 = stablehlo.select %434, %436, %432 : tensor<128xi1>, tensor<128xi32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %439 = stablehlo.concatenate %431, %438, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %439 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %200 = "mhlo.fusion"(%arg101, %arg102, %194, %189, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<16x4096xf32>, %arg211: tensor<16x4096x64xf32>, %arg212: tensor<16x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x4096xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x4096xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x4096x64xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x4096xf32>) -> tensor<16x4096x64xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x4096x64xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<16x4096x64xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x4096x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x4096x64xf32>) -> tensor<16x64x64x64xf32>
      mhlo.return %423 : tensor<16x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<16x4096xf32>, tensor<16x4096x64xf32>, tensor<16x4096xf32>) -> tensor<16x64x64x64xf32>
    %201 = "mhlo.fusion"(%arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x64x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,160]{1,0,2,3}"} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
      %414 = mhlo.copy %413 : tensor<3x3x64x160xf32>
      mhlo.return %414 : tensor<3x3x64x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
    %202 = stablehlo.convolution(%200, %201) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x64x64x64xf32>, tensor<3x3x64x160xf32>) -> tensor<16x32x32x160xf32>
    %203 = "mhlo.fusion"(%202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x32x32x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      mhlo.return %419 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %204 = "mhlo.fusion"(%203, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %205 = "mhlo.fusion"(%204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>, %arg209: tensor<16x32x32x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x1024x160xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x1024x160xf32>
      mhlo.return %424 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %206 = "mhlo.fusion"(%205, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %207 = "mhlo.fusion"(%206, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %209 = "mhlo.fusion"(%arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024xf32>, %arg212: tensor<16x32x32x160xf32>, %arg213: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x1024x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x1024x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x1024x160xf32>
      mhlo.return %429 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %210 = "mhlo.fusion"(%209, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %211 = "mhlo.fusion"(%210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<16x1024xf32>, %arg212: tensor<16x1024xf32>, %arg213: tensor<16x32x32x160xf32>, %arg214: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg213, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %421 = stablehlo.multiply %arg212, %420 : tensor<16x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x1024x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x1024x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x1024x160xf32>
      %430 = stablehlo.multiply %arg208, %420 : tensor<16x1024xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x1024x160xf32>
      %433 = stablehlo.multiply %432, %432 : tensor<16x1024x160xf32>
      mhlo.return %433 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %212 = "mhlo.fusion"(%211, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %213 = "mhlo.fusion"(%212, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %214 = "mhlo.fusion"(%213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %215 = "mhlo.fusion"(%arg108, %arg109, %214, %210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<16x1024xf32>, %arg215: tensor<16x1024xf32>, %arg216: tensor<16x32x32x160xf32>, %arg217: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x1024x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x1024x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x1024x160xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x1024xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x1024x160xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x1024x160xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x1024x160xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x1024x160xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %439 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16384x160xf32>
    %216 = stablehlo.dot_general %215, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %217 = "mhlo.fusion"(%216, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,1024,32]{3,1,2,0}"} : (tensor<16x1024x5x32xf32>) -> tensor<16x5x1024x32xf32>
      %417 = mhlo.copy %416 : tensor<16x5x1024x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x1024x32xf32>) -> tensor<80x1024x32xf32>
      mhlo.return %418 : tensor<80x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<80x1024x32xf32>
    %218 = "mhlo.fusion"(%arg108, %arg109, %214, %210, %arg106, %arg107, %208, %204, %202, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<16x1024xf32>, %arg215: tensor<16x1024xf32>, %arg216: tensor<16x32x32x160xf32>, %arg217: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x32x32x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %416 = mhlo.copy %415 : tensor<16x160x32x32xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %419 = mhlo.copy %418 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x1024x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x1024x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x1024x160xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x1024xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x1024x160xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x1024x160xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x1024x160xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x1024x160xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,160,1024]{1,2,0}"} : (tensor<16x1024x160xf32>) -> tensor<16x160x1024xf32>
      %440 = mhlo.copy %439 : tensor<16x160x1024xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x160x1024xf32>) -> tensor<16x160x32x32xf32>
      %442 = stablehlo.transpose %441, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,160]{2,1,3,0}"} : (tensor<16x160x32x32xf32>) -> tensor<16x32x32x160xf32>
      %443 = mhlo.copy %442 : tensor<16x32x32x160xf32>
      mhlo.return %443 : tensor<16x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>) -> tensor<16x32x32x160xf32>
    %219 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %414 = mhlo.copy %413 : tensor<2x2x160x160xf32>
      mhlo.return %414 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %220 = stablehlo.convolution(%218, %219) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<16x16x16x160xf32>
    %221 = "mhlo.fusion"(%220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      mhlo.return %419 : tensor<16x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<16x256x160xf32>
    %222 = "mhlo.fusion"(%221, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256xf32>
    %223 = "mhlo.fusion"(%222, %220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x160xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x160xf32>
      mhlo.return %424 : tensor<16x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<16x256x160xf32>
    %224 = "mhlo.fusion"(%223, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256x5xf32>
      mhlo.return %413 : tensor<16x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256x5xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<16x256x5xf32>, tensor<f32>) -> tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %415 = stablehlo.multiply %413, %414 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %417 = stablehlo.add %415, %416 : tensor<16x256xf32>
      %418 = stablehlo.rsqrt %417 : tensor<16x256xf32>
      mhlo.return %418 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x5xf32>) -> tensor<16x256xf32>
    %226 = "mhlo.fusion"(%arg114, %arg115, %225, %222, %220, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x160xf32>, %arg213: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x256x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x256x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x160xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x160xf32>) -> tensor<4096x160xf32>
      mhlo.return %430 : tensor<4096x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<4096x160xf32>
    %227 = stablehlo.dot_general %226, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x160xf32>, tensor<160x160xf32>) -> tensor<4096x160xf32>
    %228 = "mhlo.fusion"(%227, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<4096x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x160xf32>) -> tensor<16x256x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,32,256]{2,1,3,0}"} : (tensor<16x256x5x32xf32>) -> tensor<16x5x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x5x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x32x256xf32>) -> tensor<80x32x256xf32>
      mhlo.return %418 : tensor<80x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x160xf32>, tensor<160xf32>) -> tensor<80x32x256xf32>
    %229 = stablehlo.dot_general %217, %228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<80x1024x32xf32>, tensor<80x32x256xf32>) -> tensor<80x1024x256xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<80x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<80x1024x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<80x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
      mhlo.return %415 : tensor<16x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
    %231 = "mhlo.fusion"(%230, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024x8xf32>
      mhlo.return %413 : tensor<16x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024x8xf32>
    %232 = "mhlo.fusion"(%231, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x5x1024x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x5x1024x8xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
      mhlo.return %413 : tensor<16x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x8xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
    %233 = "mhlo.fusion"(%232, %229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024xf32>, %arg209: tensor<80x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<80x1024x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<80x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x5x1024xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x5x1024xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x5x1024xf32>) -> tensor<16x5x1024x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x5x1024x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x5x1024x256xf32>
      mhlo.return %420 : tensor<16x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024xf32>, tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
    %234 = "mhlo.fusion"(%233, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
      mhlo.return %413 : tensor<16x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
    %235 = "mhlo.fusion"(%233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<16x5x1024xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x5x1024xf32>) -> tensor<16x5x1024x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x5x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x1024x256xf32>) -> tensor<80x1024x256xf32>
      mhlo.return %415 : tensor<80x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<16x5x1024xf32>) -> tensor<80x1024x256xf32>
    %236 = stablehlo.dot_general %226, %arg118, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x160xf32>, tensor<160x160xf32>) -> tensor<4096x160xf32>
    %237 = "mhlo.fusion"(%236, %arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<4096x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x160xf32>) -> tensor<16x256x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,256,32]{3,1,2,0}"} : (tensor<16x256x5x32xf32>) -> tensor<16x5x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x5x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x256x32xf32>) -> tensor<80x256x32xf32>
      mhlo.return %418 : tensor<80x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x160xf32>, tensor<160xf32>) -> tensor<80x256x32xf32>
    %238 = stablehlo.dot_general %235, %237, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<80x1024x256xf32>, tensor<80x256x32xf32>) -> tensor<80x1024x32xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<80x1024x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x32xf32>) -> tensor<16x5x1024x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,5,32]{3,1,2,0}"} : (tensor<16x5x1024x32xf32>) -> tensor<16x1024x5x32xf32>
      %415 = mhlo.copy %414 : tensor<16x1024x5x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x5x32xf32>) -> tensor<16384x160xf32>
      mhlo.return %416 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<80x1024x32xf32>) -> tensor<16384x160xf32>
    %240 = stablehlo.dot_general %239, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %241 = "mhlo.fusion"(%240, %arg106, %arg107, %208, %204, %202, %arg105, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<16x1024xf32>, %arg212: tensor<16x1024xf32>, %arg213: tensor<16x32x32x160xf32>, %arg214: tensor<160xf32>, %arg215: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<16x32x32x160xf32>
      %417 = stablehlo.add %arg213, %416 : tensor<16x32x32x160xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %419 = mhlo.copy %418 : tensor<16x160x32x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,160]{1,2,0}"} : (tensor<16x160x1024xf32>) -> tensor<16x1024x160xf32>
      %422 = mhlo.copy %421 : tensor<16x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %424 = stablehlo.multiply %arg212, %423 : tensor<16x1024xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %426 = stablehlo.subtract %422, %425 : tensor<16x1024x160xf32>
      %427 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<16x1024x160xf32>
      %429 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<16x1024x160xf32>
      %431 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %432 = stablehlo.add %430, %431 : tensor<16x1024x160xf32>
      %433 = stablehlo.add %415, %432 : tensor<16x1024x160xf32>
      mhlo.return %433 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<16x32x32x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %242 = "mhlo.fusion"(%241, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %243 = "mhlo.fusion"(%241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x1024x160xf32>
      mhlo.return %417 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
    %244 = "mhlo.fusion"(%243, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %245 = "mhlo.fusion"(%244, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %246 = "mhlo.fusion"(%245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %247 = "mhlo.fusion"(%arg122, %arg123, %246, %241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %423 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16384x160xf32>
    %248 = stablehlo.dot_general %247, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<640x160xf32>) -> tensor<16384x640xf32>
    %249 = "mhlo.fusion"(%248, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x640xf32>, %arg209: tensor<640xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<16384x640xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<16384x640xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x640xf32>) -> tensor<16x1024x640xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,640,1024]{1,2,0}"} : (tensor<16x1024x640xf32>) -> tensor<16x640x1024xf32>
      %417 = mhlo.copy %416 : tensor<16x640x1024xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x640x1024xf32>) -> tensor<16x640x32x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,640]{2,1,3,0}"} : (tensor<16x640x32x32xf32>) -> tensor<16x32x32x640xf32>
      %420 = mhlo.copy %419 : tensor<16x32x32x640xf32>
      mhlo.return %420 : tensor<16x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x640xf32>, tensor<640xf32>) -> tensor<16x32x32x640xf32>
    %250 = "mhlo.fusion"(%arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x640xf32>
      mhlo.return %414 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %251 = stablehlo.convolution(%249, %250) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<16x32x32x640xf32>
    %252 = "mhlo.fusion"(%251, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x32x32x640xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<16x32x32x640xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x32x32x640xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x32x32x640xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,640,32,32]{1,3,2,0}"} : (tensor<16x32x32x640xf32>) -> tensor<16x640x32x32xf32>
      %418 = mhlo.copy %417 : tensor<16x640x32x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x640x32x32xf32>) -> tensor<16x640x1024xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,640]{1,2,0}"} : (tensor<16x640x1024xf32>) -> tensor<16x1024x640xf32>
      %421 = mhlo.copy %420 : tensor<16x1024x640xf32>
      %422 = stablehlo.negate %415 : tensor<16x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x32x32x640xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x32x32x640xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,640,32,32]{1,3,2,0}"} : (tensor<16x32x32x640xf32>) -> tensor<16x640x32x32xf32>
      %426 = mhlo.copy %425 : tensor<16x640x32x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x640x32x32xf32>) -> tensor<16x640x1024xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,640]{1,2,0}"} : (tensor<16x640x1024xf32>) -> tensor<16x1024x640xf32>
      %429 = mhlo.copy %428 : tensor<16x1024x640xf32>
      %430 = stablehlo.abs %429 : tensor<16x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x1024x640xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x1024x640xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x1024x640xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x1024x640xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x1024x640xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x1024x640xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x1024x640xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x1024x640xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %458 = stablehlo.negate %434 : tensor<16x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %461 = stablehlo.exponential %458 : tensor<16x1024x640xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x1024x640xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x1024x640xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x1024x640xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x1024x640xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x1024x640xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x1024x640xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x1024x640xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x1024x640xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x1024x640xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x1024x640xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x1024x640xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x1024x640xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x1024x640xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x1024x640xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x1024x640xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x1024x640xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x1024x640xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x1024x640xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x1024x640xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x1024x640xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x640xf32>) -> tensor<16384x640xf32>
      mhlo.return %520 : tensor<16384x640xf32>
    }) {output_operand_aliasing = []} : (tensor<16x32x32x640xf32>, tensor<640xf32>) -> tensor<16384x640xf32>
    %253 = stablehlo.dot_general %252, %arg128, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x640xf32>, tensor<160x640xf32>) -> tensor<16384x160xf32>
    %254 = "mhlo.fusion"(%241, %253, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16384x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x1024x160xf32>
      mhlo.return %416 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16384x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %255 = "mhlo.fusion"(%254, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %256 = "mhlo.fusion"(%254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x1024x160xf32>
      mhlo.return %417 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
    %257 = "mhlo.fusion"(%256, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %258 = "mhlo.fusion"(%257, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %260 = "mhlo.fusion"(%arg130, %arg131, %259, %254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %423 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16384x160xf32>
    %261 = stablehlo.dot_general %260, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %262 = "mhlo.fusion"(%261, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,1024,32]{3,1,2,0}"} : (tensor<16x1024x5x32xf32>) -> tensor<16x5x1024x32xf32>
      %417 = mhlo.copy %416 : tensor<16x5x1024x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x1024x32xf32>) -> tensor<80x1024x32xf32>
      mhlo.return %418 : tensor<80x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<80x1024x32xf32>
    %263 = "mhlo.fusion"(%arg130, %arg131, %259, %254, %255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,160,1024]{1,2,0}"} : (tensor<16x1024x160xf32>) -> tensor<16x160x1024xf32>
      %424 = mhlo.copy %423 : tensor<16x160x1024xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x160x1024xf32>) -> tensor<16x160x32x32xf32>
      %426 = stablehlo.transpose %425, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,160]{2,1,3,0}"} : (tensor<16x160x32x32xf32>) -> tensor<16x32x32x160xf32>
      %427 = mhlo.copy %426 : tensor<16x32x32x160xf32>
      mhlo.return %427 : tensor<16x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x32x32x160xf32>
    %264 = "mhlo.fusion"(%arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %414 = mhlo.copy %413 : tensor<2x2x160x160xf32>
      mhlo.return %414 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %265 = stablehlo.convolution(%263, %264) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<16x16x16x160xf32>
    %266 = "mhlo.fusion"(%265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      mhlo.return %419 : tensor<16x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<16x256x160xf32>
    %267 = "mhlo.fusion"(%266, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256xf32>
    %268 = "mhlo.fusion"(%267, %265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x160xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x160xf32>
      mhlo.return %424 : tensor<16x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<16x256x160xf32>
    %269 = "mhlo.fusion"(%268, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256x5xf32>
      mhlo.return %413 : tensor<16x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x160xf32>, tensor<f32>) -> tensor<16x256x5xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x5xf32>):
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.reduce(%arg208 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<16x256x5xf32>, tensor<f32>) -> tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %415 = stablehlo.multiply %413, %414 : tensor<16x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %417 = stablehlo.add %415, %416 : tensor<16x256xf32>
      %418 = stablehlo.rsqrt %417 : tensor<16x256xf32>
      mhlo.return %418 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x5xf32>) -> tensor<16x256xf32>
    %271 = "mhlo.fusion"(%arg136, %arg137, %270, %267, %265, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x160xf32>, %arg213: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<16x16x16x160xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x160xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,16,16]{1,3,2,0}"} : (tensor<16x16x16x160xf32>) -> tensor<16x160x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x160x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x16x16xf32>) -> tensor<16x160x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,160]{1,2,0}"} : (tensor<16x160x256xf32>) -> tensor<16x256x160xf32>
      %419 = mhlo.copy %418 : tensor<16x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x160xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x160xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x160xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x256x160xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x256x160xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x160xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x160xf32>) -> tensor<4096x160xf32>
      mhlo.return %430 : tensor<4096x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x160xf32>, tensor<160xf32>) -> tensor<4096x160xf32>
    %272 = stablehlo.dot_general %271, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x160xf32>, tensor<160x160xf32>) -> tensor<4096x160xf32>
    %273 = "mhlo.fusion"(%272, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<4096x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x160xf32>) -> tensor<16x256x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,32,256]{2,1,3,0}"} : (tensor<16x256x5x32xf32>) -> tensor<16x5x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x5x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x32x256xf32>) -> tensor<80x32x256xf32>
      mhlo.return %418 : tensor<80x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x160xf32>, tensor<160xf32>) -> tensor<80x32x256xf32>
    %274 = stablehlo.dot_general %262, %273, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<80x1024x32xf32>, tensor<80x32x256xf32>) -> tensor<80x1024x256xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<80x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<80x1024x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<80x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
      mhlo.return %415 : tensor<16x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024x8xf32>
      mhlo.return %413 : tensor<16x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024x8xf32>
    %277 = "mhlo.fusion"(%276, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x5x1024x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x5x1024x8xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
      mhlo.return %413 : tensor<16x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x8xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
    %278 = "mhlo.fusion"(%277, %274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024xf32>, %arg209: tensor<80x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<80x1024x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<80x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x5x1024xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x5x1024xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x5x1024xf32>) -> tensor<16x5x1024x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x5x1024x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x5x1024x256xf32>
      mhlo.return %420 : tensor<16x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024xf32>, tensor<80x1024x256xf32>) -> tensor<16x5x1024x256xf32>
    %279 = "mhlo.fusion"(%278, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
      mhlo.return %413 : tensor<16x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<f32>) -> tensor<16x5x1024xf32>
    %280 = "mhlo.fusion"(%278, %279) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x5x1024x256xf32>, %arg209: tensor<16x5x1024xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x5x1024xf32>) -> tensor<16x5x1024x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x5x1024x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x1024x256xf32>) -> tensor<80x1024x256xf32>
      mhlo.return %415 : tensor<80x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x5x1024x256xf32>, tensor<16x5x1024xf32>) -> tensor<80x1024x256xf32>
    %281 = stablehlo.dot_general %271, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x160xf32>, tensor<160x160xf32>) -> tensor<4096x160xf32>
    %282 = "mhlo.fusion"(%281, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x160xf32>, %arg209: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<4096x160xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x160xf32>) -> tensor<16x256x5x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,5,256,32]{3,1,2,0}"} : (tensor<16x256x5x32xf32>) -> tensor<16x5x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x5x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x5x256x32xf32>) -> tensor<80x256x32xf32>
      mhlo.return %418 : tensor<80x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x160xf32>, tensor<160xf32>) -> tensor<80x256x32xf32>
    %283 = stablehlo.dot_general %280, %282, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<80x1024x256xf32>, tensor<80x256x32xf32>) -> tensor<80x1024x32xf32>
    %284 = "mhlo.fusion"(%283) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<80x1024x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<80x1024x32xf32>) -> tensor<16x5x1024x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,5,32]{3,1,2,0}"} : (tensor<16x5x1024x32xf32>) -> tensor<16x1024x5x32xf32>
      %415 = mhlo.copy %414 : tensor<16x1024x5x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x5x32xf32>) -> tensor<16384x160xf32>
      mhlo.return %416 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<80x1024x32xf32>) -> tensor<16384x160xf32>
    %285 = stablehlo.dot_general %284, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %286 = "mhlo.fusion"(%254, %285, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16384x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x1024x160xf32>
      mhlo.return %416 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16384x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %287 = "mhlo.fusion"(%286, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %288 = "mhlo.fusion"(%286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x1024x160xf32>
      mhlo.return %417 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
    %289 = "mhlo.fusion"(%288, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %290 = "mhlo.fusion"(%289, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %292 = "mhlo.fusion"(%arg144, %arg145, %291, %286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %423 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16384x160xf32>
    %293 = stablehlo.dot_general %292, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<640x160xf32>) -> tensor<16384x640xf32>
    %294 = "mhlo.fusion"(%293, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x640xf32>, %arg209: tensor<640xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<16384x640xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<16384x640xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x640xf32>) -> tensor<16x1024x640xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,640,1024]{1,2,0}"} : (tensor<16x1024x640xf32>) -> tensor<16x640x1024xf32>
      %417 = mhlo.copy %416 : tensor<16x640x1024xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x640x1024xf32>) -> tensor<16x640x32x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,32,32,640]{2,1,3,0}"} : (tensor<16x640x32x32xf32>) -> tensor<16x32x32x640xf32>
      %420 = mhlo.copy %419 : tensor<16x32x32x640xf32>
      mhlo.return %420 : tensor<16x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x640xf32>, tensor<640xf32>) -> tensor<16x32x32x640xf32>
    %295 = "mhlo.fusion"(%arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x640xf32>
      mhlo.return %414 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %296 = stablehlo.convolution(%294, %295) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<16x32x32x640xf32>
    %297 = "mhlo.fusion"(%296, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x32x32x640xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<16x32x32x640xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x32x32x640xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x32x32x640xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,640,32,32]{1,3,2,0}"} : (tensor<16x32x32x640xf32>) -> tensor<16x640x32x32xf32>
      %418 = mhlo.copy %417 : tensor<16x640x32x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x640x32x32xf32>) -> tensor<16x640x1024xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,640]{1,2,0}"} : (tensor<16x640x1024xf32>) -> tensor<16x1024x640xf32>
      %421 = mhlo.copy %420 : tensor<16x1024x640xf32>
      %422 = stablehlo.negate %415 : tensor<16x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x32x32x640xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x32x32x640xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,640,32,32]{1,3,2,0}"} : (tensor<16x32x32x640xf32>) -> tensor<16x640x32x32xf32>
      %426 = mhlo.copy %425 : tensor<16x640x32x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x640x32x32xf32>) -> tensor<16x640x1024xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,640]{1,2,0}"} : (tensor<16x640x1024xf32>) -> tensor<16x1024x640xf32>
      %429 = mhlo.copy %428 : tensor<16x1024x640xf32>
      %430 = stablehlo.abs %429 : tensor<16x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x1024x640xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x1024x640xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x1024x640xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x1024x640xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x1024x640xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x1024x640xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x1024x640xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x1024x640xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %458 = stablehlo.negate %434 : tensor<16x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %461 = stablehlo.exponential %458 : tensor<16x1024x640xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x1024x640xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x1024x640xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x1024x640xf32>, tensor<16x1024x640xf32>) -> tensor<16x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x1024x640xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x1024x640xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x1024x640xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x1024x640xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x1024x640xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x1024x640xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x1024x640xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x1024x640xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x1024x640xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x1024x640xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x1024x640xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x1024x640xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x1024x640xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x1024x640xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x1024x640xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x1024x640xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x1024x640xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x1024x640xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x1024x640xi1>, tensor<16x1024x640xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x1024x640xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x640xf32>) -> tensor<16384x640xf32>
      mhlo.return %520 : tensor<16384x640xf32>
    }) {output_operand_aliasing = []} : (tensor<16x32x32x640xf32>, tensor<640xf32>) -> tensor<16384x640xf32>
    %298 = stablehlo.dot_general %297, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x640xf32>, tensor<160x640xf32>) -> tensor<16384x160xf32>
    %299 = "mhlo.fusion"(%286, %298, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16384x160xf32>, %arg210: tensor<160xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<16384x160xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x1024x160xf32>
      mhlo.return %416 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16384x160xf32>, tensor<160xf32>) -> tensor<16x1024x160xf32>
    %300 = "mhlo.fusion"(%299, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %301 = "mhlo.fusion"(%299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x1024x160xf32>
      mhlo.return %417 : tensor<16x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
    %302 = "mhlo.fusion"(%301, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024x160xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
      mhlo.return %413 : tensor<16x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x160xf32>, tensor<f32>) -> tensor<16x1024x5xf32>
    %303 = "mhlo.fusion"(%302, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x1024x5xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
      mhlo.return %413 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024x5xf32>, tensor<f32>) -> tensor<16x1024xf32>
    %304 = "mhlo.fusion"(%303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x1024xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x1024xf32>
      mhlo.return %417 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %305 = "mhlo.fusion"(%arg152, %arg153, %304, %299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16x32x32x160xf32>
      %424 = stablehlo.transpose %423, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,160,32,32]{1,3,2,0}"} : (tensor<16x32x32x160xf32>) -> tensor<16x160x32x32xf32>
      %425 = mhlo.copy %424 : tensor<16x160x32x32xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x160x32x32xf32>) -> tensor<16x160x1024xf32>
      mhlo.return %426 : tensor<16x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x160x1024xf32>
    %306 = "mhlo.fusion"(%arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>):
      %413 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[160,256]{0,1}"} : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %414 = mhlo.copy %413 : tensor<160x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<16x160x256xf32>
      mhlo.return %415 : tensor<16x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>) -> tensor<16x160x256xf32>
    %307 = stablehlo.dot_general %305, %306, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x160x1024xf32>, tensor<16x160x256xf32>) -> tensor<16x1024x256xf32>
    %308 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<15> : tensor<i32>
      %413 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %415 = stablehlo.iota dim = 0 : tensor<128xi32>
      %416 = stablehlo.convert %415 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.add %416, %417 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.add %420, %421 : tensor<128xf32>
      %423 = stablehlo.maximum %414, %422 : tensor<128xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %425 = stablehlo.convert %424 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %426 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %427 = stablehlo.add %425, %426 : tensor<128x1xi32>
      %428 = stablehlo.minimum %413, %427 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.compare LT, %428, %429 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<16> : tensor<i32>
      %431 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %432 = stablehlo.add %428, %431 : tensor<128x1xi32>
      %433 = stablehlo.select %430, %432, %428 : tensor<128x1xi1>, tensor<128x1xi32>
      %434 = mhlo.bitcast %433 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %436 = stablehlo.convert %423 : (tensor<128xf32>) -> tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %439 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %440 = stablehlo.add %436, %439 : tensor<128xi32>
      %441 = stablehlo.select %438, %440, %436 : tensor<128xi1>, tensor<128xi32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %443 = stablehlo.concatenate %435, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %443 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %309 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.iota dim = 0 : tensor<128xi32>
      %415 = stablehlo.convert %414 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %417 = stablehlo.add %415, %416 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %422 = stablehlo.maximum %413, %421 : tensor<128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %424 = stablehlo.convert %423 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.compare LT, %424, %425 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<16> : tensor<i32>
      %427 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %428 = stablehlo.add %424, %427 : tensor<128x1xi32>
      %429 = stablehlo.select %426, %428, %424 : tensor<128x1xi1>, tensor<128x1xi32>
      %430 = mhlo.bitcast %429 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %432 = stablehlo.convert %422 : (tensor<128xf32>) -> tensor<128xi32>
      %433 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.compare LT, %432, %433 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %435 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %436 = stablehlo.add %432, %435 : tensor<128xi32>
      %437 = stablehlo.select %434, %436, %432 : tensor<128xi1>, tensor<128xi32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %439 = stablehlo.concatenate %431, %438, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %439 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %310 = "mhlo.fusion"(%arg152, %arg153, %304, %299, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<16x1024xf32>, %arg211: tensor<16x1024x160xf32>, %arg212: tensor<16x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x1024xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x1024xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x1024x160xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x1024xf32>) -> tensor<16x1024x160xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x1024x160xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<16x1024x160xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x1024x160xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x160xf32>) -> tensor<16x32x32x160xf32>
      mhlo.return %423 : tensor<16x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<16x1024xf32>, tensor<16x1024x160xf32>, tensor<16x1024xf32>) -> tensor<16x32x32x160xf32>
    %311 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,160,256]{1,0,2,3}"} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
      %414 = mhlo.copy %413 : tensor<3x3x160x256xf32>
      mhlo.return %414 : tensor<3x3x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
    %312 = stablehlo.convolution(%310, %311) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x32x32x160xf32>, tensor<3x3x160x256xf32>) -> tensor<16x16x16x256xf32>
    %313 = "mhlo.fusion"(%312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x16x16x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x256x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %419 = mhlo.copy %418 : tensor<16x256x256xf32>
      mhlo.return %419 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %314 = "mhlo.fusion"(%313, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %315 = "mhlo.fusion"(%314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<16x16x16x256xf32>, %arg210: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %414 = stablehlo.add %arg209, %413 : tensor<16x16x16x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x256x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %419 = mhlo.copy %418 : tensor<16x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x256xf32>
      %424 = stablehlo.multiply %423, %423 : tensor<16x256x256xf32>
      mhlo.return %424 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<16x16x16x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %316 = "mhlo.fusion"(%315, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %317 = "mhlo.fusion"(%316, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %318 = "mhlo.fusion"(%317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %319 = "mhlo.fusion"(%arg157, %arg158, %318, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x16x16x256xf32>, %arg213: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %414 = stablehlo.add %arg212, %413 : tensor<16x16x16x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x256x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %419 = mhlo.copy %418 : tensor<16x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x256xf32>
      %424 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x256xf32>
      %426 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x256xf32>
      mhlo.return %429 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %320 = "mhlo.fusion"(%319, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %321 = "mhlo.fusion"(%320, %arg157, %arg158, %318, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x256xf32>, %arg213: tensor<16x16x16x256xf32>, %arg214: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %414 = stablehlo.add %arg213, %413 : tensor<16x16x16x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x256x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %419 = mhlo.copy %418 : tensor<16x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg212, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x256xf32>
      %424 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x256xf32>
      %426 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x256xf32>
      %430 = stablehlo.multiply %arg208, %420 : tensor<16x256xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x256x256xf32>
      %433 = stablehlo.multiply %432, %432 : tensor<16x256x256xf32>
      mhlo.return %433 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %322 = "mhlo.fusion"(%321, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %323 = "mhlo.fusion"(%322, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %324 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %325 = "mhlo.fusion"(%arg159, %arg160, %324, %320, %arg157, %arg158, %318, %314, %312, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<16x256xf32>, %arg215: tensor<16x256xf32>, %arg216: tensor<16x16x16x256xf32>, %arg217: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %414 = stablehlo.add %arg216, %413 : tensor<16x16x16x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %416 = mhlo.copy %415 : tensor<16x256x16x16xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %419 = mhlo.copy %418 : tensor<16x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %421 = stablehlo.multiply %arg215, %420 : tensor<16x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<16x256x256xf32>
      %424 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %425 = stablehlo.multiply %423, %424 : tensor<16x256x256xf32>
      %426 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<16x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %429 = stablehlo.add %427, %428 : tensor<16x256x256xf32>
      %430 = stablehlo.multiply %arg211, %420 : tensor<16x256xf32>
      %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %432 = stablehlo.subtract %429, %431 : tensor<16x256x256xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<16x256x256xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %436 = stablehlo.multiply %434, %435 : tensor<16x256x256xf32>
      %437 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %438 = stablehlo.add %436, %437 : tensor<16x256x256xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %439 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x256xf32>, tensor<256xf32>) -> tensor<4096x256xf32>
    %326 = stablehlo.dot_general %325, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %327 = "mhlo.fusion"(%326, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,256,32]{3,1,2,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x8x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %418 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x256x32xf32>
    %328 = stablehlo.dot_general %325, %arg163, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %329 = "mhlo.fusion"(%328, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,32,256]{2,1,3,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x8x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x32x256xf32>) -> tensor<128x32x256xf32>
      mhlo.return %418 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x32x256xf32>
    %330 = stablehlo.dot_general %327, %329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x32xf32>, tensor<128x32x256xf32>) -> tensor<128x256x256xf32>
    %331 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<128x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
      mhlo.return %415 : tensor<16x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
    %332 = "mhlo.fusion"(%331, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256x8xf32>
      mhlo.return %413 : tensor<16x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256x8xf32>
    %333 = "mhlo.fusion"(%332, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x8x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x8x256x8xf32>, tensor<f32>) -> tensor<16x8x256xf32>
      mhlo.return %413 : tensor<16x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x8xf32>, tensor<f32>) -> tensor<16x8x256xf32>
    %334 = "mhlo.fusion"(%333, %330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256xf32>, %arg209: tensor<128x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<128x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x8x256xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x8x256xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x8x256xf32>) -> tensor<16x8x256x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x8x256x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x8x256x256xf32>
      mhlo.return %420 : tensor<16x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256xf32>, tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
    %335 = "mhlo.fusion"(%334, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256xf32>
      mhlo.return %413 : tensor<16x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256xf32>
    %336 = "mhlo.fusion"(%334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<16x8x256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x8x256xf32>) -> tensor<16x8x256x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x8x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x256xf32>) -> tensor<128x256x256xf32>
      mhlo.return %415 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<16x8x256xf32>) -> tensor<128x256x256xf32>
    %337 = stablehlo.dot_general %325, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %338 = "mhlo.fusion"(%337, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,256,32]{3,1,2,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x8x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %418 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x256x32xf32>
    %339 = stablehlo.dot_general %336, %338, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x256xf32>, tensor<128x256x32xf32>) -> tensor<128x256x32xf32>
    %340 = "mhlo.fusion"(%339) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<16x8x256x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,8,32]{3,1,2,0}"} : (tensor<16x8x256x32xf32>) -> tensor<16x256x8x32xf32>
      %415 = mhlo.copy %414 : tensor<16x256x8x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x8x32xf32>) -> tensor<4096x256xf32>
      mhlo.return %416 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x32xf32>) -> tensor<4096x256xf32>
    %341 = stablehlo.dot_general %340, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %342 = "mhlo.fusion"(%341, %arg157, %arg158, %318, %314, %312, %arg156, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<16x256xf32>, %arg212: tensor<16x256xf32>, %arg213: tensor<16x16x16x256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<16x16x16x256xf32>
      %417 = stablehlo.add %arg213, %416 : tensor<16x16x16x256xf32>
      %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %419 = mhlo.copy %418 : tensor<16x256x16x16xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %422 = mhlo.copy %421 : tensor<16x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %424 = stablehlo.multiply %arg212, %423 : tensor<16x256xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %426 = stablehlo.subtract %422, %425 : tensor<16x256x256xf32>
      %427 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<16x256x256xf32>
      %429 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %430 = stablehlo.multiply %428, %429 : tensor<16x256x256xf32>
      %431 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %432 = stablehlo.add %430, %431 : tensor<16x256x256xf32>
      %433 = stablehlo.add %415, %432 : tensor<16x256x256xf32>
      mhlo.return %433 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256xf32>, tensor<16x16x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %343 = "mhlo.fusion"(%342, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %344 = "mhlo.fusion"(%342, %343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x256x256xf32>
      mhlo.return %417 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<16x256x256xf32>
    %345 = "mhlo.fusion"(%344, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %346 = "mhlo.fusion"(%345, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %348 = "mhlo.fusion"(%arg169, %arg170, %347, %342, %343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256x256xf32>, %arg212: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x256x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x256x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %423 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<4096x256xf32>
    %349 = stablehlo.dot_general %348, %arg171, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<1024x256xf32>) -> tensor<4096x1024xf32>
    %350 = "mhlo.fusion"(%349, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x1024xf32>, %arg209: tensor<1024xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<4096x1024xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x1024xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x1024xf32>) -> tensor<16x256x1024xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,256]{1,2,0}"} : (tensor<16x256x1024xf32>) -> tensor<16x1024x256xf32>
      %417 = mhlo.copy %416 : tensor<16x1024x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x256xf32>) -> tensor<16x1024x16x16xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,1024]{2,1,3,0}"} : (tensor<16x1024x16x16xf32>) -> tensor<16x16x16x1024xf32>
      %420 = mhlo.copy %419 : tensor<16x16x16x1024xf32>
      mhlo.return %420 : tensor<16x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x1024xf32>, tensor<1024xf32>) -> tensor<16x16x16x1024xf32>
    %351 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x1024xf32>
      mhlo.return %414 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %352 = stablehlo.convolution(%350, %351) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<16x16x16x1024xf32>
    %353 = "mhlo.fusion"(%352, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16x16x1024xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<16x16x16x1024xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x16x16x1024xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x16x16x1024xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,16,16]{1,3,2,0}"} : (tensor<16x16x16x1024xf32>) -> tensor<16x1024x16x16xf32>
      %418 = mhlo.copy %417 : tensor<16x1024x16x16xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x256xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,1024]{1,2,0}"} : (tensor<16x1024x256xf32>) -> tensor<16x256x1024xf32>
      %421 = mhlo.copy %420 : tensor<16x256x1024xf32>
      %422 = stablehlo.negate %415 : tensor<16x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x16x16x1024xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x16x16x1024xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,16,16]{1,3,2,0}"} : (tensor<16x16x16x1024xf32>) -> tensor<16x1024x16x16xf32>
      %426 = mhlo.copy %425 : tensor<16x1024x16x16xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x256xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,1024]{1,2,0}"} : (tensor<16x1024x256xf32>) -> tensor<16x256x1024xf32>
      %429 = mhlo.copy %428 : tensor<16x256x1024xf32>
      %430 = stablehlo.abs %429 : tensor<16x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x256x1024xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x256x1024xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x256x1024xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x256x1024xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x256x1024xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x256x1024xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x256x1024xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x256x1024xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %458 = stablehlo.negate %434 : tensor<16x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %461 = stablehlo.exponential %458 : tensor<16x256x1024xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x256x1024xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x256x1024xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x256x1024xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x256x1024xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x256x1024xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x256x1024xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x256x1024xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x256x1024xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x256x1024xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x256x1024xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x256x1024xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x256x1024xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x256x1024xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x256x1024xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x256x1024xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x256x1024xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x256x1024xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x256x1024xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x256x1024xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x256x1024xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x1024xf32>) -> tensor<4096x1024xf32>
      mhlo.return %520 : tensor<4096x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x1024xf32>, tensor<1024xf32>) -> tensor<4096x1024xf32>
    %354 = stablehlo.dot_general %353, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x1024xf32>, tensor<256x1024xf32>) -> tensor<4096x256xf32>
    %355 = "mhlo.fusion"(%342, %354, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<4096x256xf32>, %arg210: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x256x256xf32>
      mhlo.return %416 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<4096x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %356 = "mhlo.fusion"(%355, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %357 = "mhlo.fusion"(%355, %356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x256x256xf32>
      mhlo.return %417 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<16x256x256xf32>
    %358 = "mhlo.fusion"(%357, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %359 = "mhlo.fusion"(%358, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %360 = "mhlo.fusion"(%359) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %361 = "mhlo.fusion"(%arg177, %arg178, %360, %355, %356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256x256xf32>, %arg212: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x256x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x256x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %423 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<4096x256xf32>
    %362 = stablehlo.dot_general %361, %arg179, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %363 = "mhlo.fusion"(%362, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,256,32]{3,1,2,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x8x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %418 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x256x32xf32>
    %364 = stablehlo.dot_general %361, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %365 = "mhlo.fusion"(%364, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,32,256]{2,1,3,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x32x256xf32>
      %417 = mhlo.copy %416 : tensor<16x8x32x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x32x256xf32>) -> tensor<128x32x256xf32>
      mhlo.return %418 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x32x256xf32>
    %366 = stablehlo.dot_general %363, %365, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x32xf32>, tensor<128x32x256xf32>) -> tensor<128x256x256xf32>
    %367 = "mhlo.fusion"(%366) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<128x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
      mhlo.return %415 : tensor<16x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
    %368 = "mhlo.fusion"(%367, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256x8xf32>
      mhlo.return %413 : tensor<16x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256x8xf32>
    %369 = "mhlo.fusion"(%368, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x8x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<16x8x256x8xf32>, tensor<f32>) -> tensor<16x8x256xf32>
      mhlo.return %413 : tensor<16x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x8xf32>, tensor<f32>) -> tensor<16x8x256xf32>
    %370 = "mhlo.fusion"(%369, %366) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256xf32>, %arg209: tensor<128x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x256x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<128x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x8x256xf32>
      %417 = stablehlo.maximum %416, %arg208 : tensor<16x8x256xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<16x8x256xf32>) -> tensor<16x8x256x256xf32>
      %419 = stablehlo.subtract %415, %418 : tensor<16x8x256x256xf32>
      %420 = stablehlo.exponential %419 : tensor<16x8x256x256xf32>
      mhlo.return %420 : tensor<16x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256xf32>, tensor<128x256x256xf32>) -> tensor<16x8x256x256xf32>
    %371 = "mhlo.fusion"(%370, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256xf32>
      mhlo.return %413 : tensor<16x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<f32>) -> tensor<16x8x256xf32>
    %372 = "mhlo.fusion"(%370, %371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x8x256x256xf32>, %arg209: tensor<16x8x256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<16x8x256xf32>) -> tensor<16x8x256x256xf32>
      %414 = stablehlo.divide %arg208, %413 : tensor<16x8x256x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x256xf32>) -> tensor<128x256x256xf32>
      mhlo.return %415 : tensor<128x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x8x256x256xf32>, tensor<16x8x256xf32>) -> tensor<128x256x256xf32>
    %373 = stablehlo.dot_general %361, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %374 = "mhlo.fusion"(%373, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x256xf32>, %arg209: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x8x32xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,256,32]{3,1,2,0}"} : (tensor<16x256x8x32xf32>) -> tensor<16x8x256x32xf32>
      %417 = mhlo.copy %416 : tensor<16x8x256x32xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x8x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %418 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x256xf32>, tensor<256xf32>) -> tensor<128x256x32xf32>
    %375 = stablehlo.dot_general %372, %374, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x256x256xf32>, tensor<128x256x32xf32>) -> tensor<128x256x32xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x256x32xf32>):
      %413 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x256x32xf32>) -> tensor<16x8x256x32xf32>
      %414 = stablehlo.transpose %413, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,8,32]{3,1,2,0}"} : (tensor<16x8x256x32xf32>) -> tensor<16x256x8x32xf32>
      %415 = mhlo.copy %414 : tensor<16x256x8x32xf32>
      %416 = mhlo.bitcast %415 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x8x32xf32>) -> tensor<4096x256xf32>
      mhlo.return %416 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x32xf32>) -> tensor<4096x256xf32>
    %377 = stablehlo.dot_general %376, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x256xf32>) -> tensor<4096x256xf32>
    %378 = "mhlo.fusion"(%355, %377, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<4096x256xf32>, %arg210: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x256x256xf32>
      mhlo.return %416 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<4096x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %379 = "mhlo.fusion"(%378, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %380 = "mhlo.fusion"(%378, %379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x256x256xf32>
      mhlo.return %417 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<16x256x256xf32>
    %381 = "mhlo.fusion"(%380, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %382 = "mhlo.fusion"(%381, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %383 = "mhlo.fusion"(%382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %384 = "mhlo.fusion"(%arg187, %arg188, %383, %378, %379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256x256xf32>, %arg212: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x256x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x256x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<4096x256xf32>
      mhlo.return %423 : tensor<4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<4096x256xf32>
    %385 = stablehlo.dot_general %384, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<1024x256xf32>) -> tensor<4096x1024xf32>
    %386 = "mhlo.fusion"(%385, %arg190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<4096x1024xf32>, %arg209: tensor<1024xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<4096x1024xf32>
      %414 = stablehlo.add %413, %arg208 : tensor<4096x1024xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x1024xf32>) -> tensor<16x256x1024xf32>
      %416 = stablehlo.transpose %415, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,1024,256]{1,2,0}"} : (tensor<16x256x1024xf32>) -> tensor<16x1024x256xf32>
      %417 = mhlo.copy %416 : tensor<16x1024x256xf32>
      %418 = mhlo.bitcast %417 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x1024x256xf32>) -> tensor<16x1024x16x16xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,16,16,1024]{2,1,3,0}"} : (tensor<16x1024x16x16xf32>) -> tensor<16x16x16x1024xf32>
      %420 = mhlo.copy %419 : tensor<16x16x16x1024xf32>
      mhlo.return %420 : tensor<16x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<4096x1024xf32>, tensor<1024xf32>) -> tensor<16x16x16x1024xf32>
    %387 = "mhlo.fusion"(%arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %414 = mhlo.copy %413 : tensor<3x3x1x1024xf32>
      mhlo.return %414 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %388 = stablehlo.convolution(%386, %387) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<16x16x16x1024xf32>
    %389 = "mhlo.fusion"(%388, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x16x16x1024xf32>
      %414 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<16x16x16x1024xf32>
      %415 = stablehlo.add %arg208, %414 : tensor<16x16x16x1024xf32>
      %416 = stablehlo.multiply %413, %415 : tensor<16x16x16x1024xf32>
      %417 = stablehlo.transpose %416, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,16,16]{1,3,2,0}"} : (tensor<16x16x16x1024xf32>) -> tensor<16x1024x16x16xf32>
      %418 = mhlo.copy %417 : tensor<16x1024x16x16xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x256xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,1024]{1,2,0}"} : (tensor<16x1024x256xf32>) -> tensor<16x256x1024xf32>
      %421 = mhlo.copy %420 : tensor<16x256x1024xf32>
      %422 = stablehlo.negate %415 : tensor<16x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x16x16x1024xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<16x16x16x1024xf32>
      %425 = stablehlo.transpose %424, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,1024,16,16]{1,3,2,0}"} : (tensor<16x16x16x1024xf32>) -> tensor<16x1024x16x16xf32>
      %426 = mhlo.copy %425 : tensor<16x1024x16x16xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1024x16x16xf32>) -> tensor<16x1024x256xf32>
      %428 = stablehlo.transpose %427, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,1024]{1,2,0}"} : (tensor<16x1024x256xf32>) -> tensor<16x256x1024xf32>
      %429 = mhlo.copy %428 : tensor<16x256x1024xf32>
      %430 = stablehlo.abs %429 : tensor<16x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %434 = stablehlo.multiply %429, %429 : tensor<16x256x1024xf32>
      %435 = stablehlo.multiply %433, %434 : tensor<16x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %437 = stablehlo.add %435, %436 : tensor<16x256x1024xf32>
      %438 = stablehlo.multiply %437, %434 : tensor<16x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %440 = stablehlo.add %438, %439 : tensor<16x256x1024xf32>
      %441 = stablehlo.multiply %440, %434 : tensor<16x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %443 = stablehlo.add %441, %442 : tensor<16x256x1024xf32>
      %444 = stablehlo.multiply %443, %434 : tensor<16x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %446 = stablehlo.add %444, %445 : tensor<16x256x1024xf32>
      %447 = stablehlo.multiply %446, %434 : tensor<16x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %449 = stablehlo.add %447, %448 : tensor<16x256x1024xf32>
      %450 = stablehlo.multiply %449, %434 : tensor<16x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %452 = stablehlo.add %450, %451 : tensor<16x256x1024xf32>
      %453 = stablehlo.multiply %429, %452 : tensor<16x256x1024xf32>
      %454 = stablehlo.subtract %431, %453 : tensor<16x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %456 = stablehlo.compare LT, %429, %455 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %458 = stablehlo.negate %434 : tensor<16x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %460 = stablehlo.compare LT, %458, %459 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %461 = stablehlo.exponential %458 : tensor<16x256x1024xf32>
      %462 = stablehlo.divide %431, %430 : tensor<16x256x1024xf32>
      %463 = stablehlo.multiply %461, %462 : tensor<16x256x1024xf32>
      %464 = stablehlo.compare LT, %430, %457 : (tensor<16x256x1024xf32>, tensor<16x256x1024xf32>) -> tensor<16x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %466 = stablehlo.divide %431, %434 : tensor<16x256x1024xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<16x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %469 = stablehlo.add %467, %468 : tensor<16x256x1024xf32>
      %470 = stablehlo.multiply %469, %466 : tensor<16x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %472 = stablehlo.add %470, %471 : tensor<16x256x1024xf32>
      %473 = stablehlo.multiply %472, %466 : tensor<16x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %475 = stablehlo.add %473, %474 : tensor<16x256x1024xf32>
      %476 = stablehlo.multiply %475, %466 : tensor<16x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %478 = stablehlo.add %476, %477 : tensor<16x256x1024xf32>
      %479 = stablehlo.multiply %478, %466 : tensor<16x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %481 = stablehlo.add %479, %480 : tensor<16x256x1024xf32>
      %482 = stablehlo.multiply %481, %466 : tensor<16x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %484 = stablehlo.add %482, %483 : tensor<16x256x1024xf32>
      %485 = stablehlo.multiply %484, %466 : tensor<16x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %487 = stablehlo.add %485, %486 : tensor<16x256x1024xf32>
      %488 = stablehlo.multiply %487, %466 : tensor<16x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %489 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %490 = stablehlo.add %488, %489 : tensor<16x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %492 = stablehlo.multiply %491, %466 : tensor<16x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<16x256x1024xf32>
      %495 = stablehlo.multiply %494, %466 : tensor<16x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %497 = stablehlo.add %495, %496 : tensor<16x256x1024xf32>
      %498 = stablehlo.multiply %497, %466 : tensor<16x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %499 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %500 = stablehlo.add %498, %499 : tensor<16x256x1024xf32>
      %501 = stablehlo.multiply %500, %466 : tensor<16x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %502 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %503 = stablehlo.add %501, %502 : tensor<16x256x1024xf32>
      %504 = stablehlo.multiply %503, %466 : tensor<16x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %506 = stablehlo.add %504, %505 : tensor<16x256x1024xf32>
      %507 = stablehlo.multiply %506, %466 : tensor<16x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %508 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %509 = stablehlo.add %507, %508 : tensor<16x256x1024xf32>
      %510 = stablehlo.multiply %509, %466 : tensor<16x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<16x256x1024xf32>
      %512 = stablehlo.add %510, %511 : tensor<16x256x1024xf32>
      %513 = stablehlo.select %464, %490, %512 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %514 = stablehlo.multiply %463, %513 : tensor<16x256x1024xf32>
      %515 = stablehlo.select %460, %455, %514 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %516 = stablehlo.subtract %457, %515 : tensor<16x256x1024xf32>
      %517 = stablehlo.select %456, %516, %515 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %518 = stablehlo.select %432, %454, %517 : tensor<16x256x1024xi1>, tensor<16x256x1024xf32>
      %519 = stablehlo.multiply %421, %518 : tensor<16x256x1024xf32>
      %520 = mhlo.bitcast %519 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x1024xf32>) -> tensor<4096x1024xf32>
      mhlo.return %520 : tensor<4096x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x16x1024xf32>, tensor<1024xf32>) -> tensor<4096x1024xf32>
    %390 = stablehlo.dot_general %389, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4096x1024xf32>, tensor<256x1024xf32>) -> tensor<4096x256xf32>
    %391 = "mhlo.fusion"(%378, %390, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<4096x256xf32>, %arg210: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<4096x256xf32>
      %414 = stablehlo.add %413, %arg209 : tensor<4096x256xf32>
      %415 = mhlo.bitcast %414 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.add %415, %arg208 : tensor<16x256x256xf32>
      mhlo.return %416 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<4096x256xf32>, tensor<256xf32>) -> tensor<16x256x256xf32>
    %392 = "mhlo.fusion"(%391, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256xf32>
    %393 = "mhlo.fusion"(%391, %392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg209, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg208, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.multiply %416, %416 : tensor<16x256x256xf32>
      mhlo.return %417 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<16x256x256xf32>
    %394 = "mhlo.fusion"(%393, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256x256xf32>, %arg209: tensor<f32>):
      %413 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %414 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %414 : tensor<f32>
      }) : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
      mhlo.return %413 : tensor<16x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x256xf32>, tensor<f32>) -> tensor<16x256x8xf32>
    %395 = "mhlo.fusion"(%394, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<16x256x8xf32>, %arg209: tensor<f32>):
      %413 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %413 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8xf32>, tensor<f32>) -> tensor<16x256xf32>
    %396 = "mhlo.fusion"(%395) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg208, %413 : tensor<16x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %416 = stablehlo.add %414, %415 : tensor<16x256xf32>
      %417 = stablehlo.rsqrt %416 : tensor<16x256xf32>
      mhlo.return %417 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %397 = "mhlo.fusion"(%arg195, %arg196, %396, %391, %392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<16x256xf32>, %arg211: tensor<16x256x256xf32>, %arg212: tensor<16x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
      %414 = stablehlo.multiply %arg212, %413 : tensor<16x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %416 = stablehlo.subtract %arg211, %415 : tensor<16x256x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<16x256xf32>) -> tensor<16x256x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x256x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<16x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %422 = stablehlo.add %420, %421 : tensor<16x256x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<16x16x16x256xf32>
      %424 = stablehlo.transpose %423, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,256,16,16]{1,3,2,0}"} : (tensor<16x16x16x256xf32>) -> tensor<16x256x16x16xf32>
      %425 = mhlo.copy %424 : tensor<16x256x16x16xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x16x16xf32>) -> tensor<16x256x256xf32>
      mhlo.return %426 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<16x256x256xf32>, tensor<16x256xf32>) -> tensor<16x256x256xf32>
    %398 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %413 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %414 = mhlo.copy %413 : tensor<256x256xf32>
      %415 = stablehlo.broadcast_in_dim %414, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<16x256x256xf32>
      mhlo.return %415 : tensor<16x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<16x256x256xf32>
    %399 = stablehlo.dot_general %397, %398, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256x256xf32>, tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
    %400 = "mhlo.fusion"(%1, %2, %86, %arg198, %87, %88, %89, %197, %arg199, %198, %199, %307, %arg200, %cst_0, %308, %309, %c, %cst_1, %399, %arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x2xi32>, %arg209: tensor<128x128x2xi32>, %arg210: tensor<16x16384x256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128x128x2xi32>, %arg213: tensor<128x128x2xi32>, %arg214: tensor<128xf32>, %arg215: tensor<16x4096x256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<128x128x2xi32>, %arg218: tensor<128x128x2xi32>, %arg219: tensor<16x1024x256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<f32>, %arg222: tensor<128x128x2xi32>, %arg223: tensor<128x128x2xi32>, %arg224: tensor<i32>, %arg225: tensor<f32>, %arg226: tensor<16x256x256xf32>, %arg227: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg227, dims = [2] : (tensor<256xf32>) -> tensor<16x256x256xf32>
      %414 = stablehlo.add %arg226, %413 : tensor<16x256x256xf32>
      %415 = stablehlo.transpose %414, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,256]{1,2,0}"} : (tensor<16x256x256xf32>) -> tensor<16x256x256xf32>
      %416 = mhlo.copy %415 : tensor<16x256x256xf32>
      %417 = mhlo.bitcast %416 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x256xf32>) -> tensor<16x256x16x16xf32>
      %418 = mhlo.bitcast %arg223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %419 = "stablehlo.gather"(%417, %418) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %421 = stablehlo.transpose %420, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %422 = mhlo.copy %421 : tensor<16x256x128x128xf32>
      %423 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %425 = stablehlo.multiply %arg214, %424 : tensor<128xf32>
      %426 = stablehlo.broadcast_in_dim %arg225, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %427 = stablehlo.add %425, %426 : tensor<128xf32>
      %428 = stablehlo.maximum %423, %427 : tensor<128xf32>
      %429 = mhlo.bitcast %428 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %430 = stablehlo.convert %429 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %431 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %433 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %434 = stablehlo.add %430, %433 : tensor<128x1xi32>
      %435 = stablehlo.select %432, %434, %430 : tensor<128x1xi1>, tensor<128x1xi32>
      %436 = mhlo.bitcast %435 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %438 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %439 = stablehlo.convert %428 : (tensor<128xf32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %440 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %441 = stablehlo.add %439, %440 : tensor<128xi32>
      %442 = stablehlo.minimum %438, %441 : tensor<128xi32>
      %443 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %444 = stablehlo.compare LT, %442, %443 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %445 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %446 = stablehlo.add %442, %445 : tensor<128xi32>
      %447 = stablehlo.select %444, %446, %442 : tensor<128xi1>, tensor<128xi32>
      %448 = stablehlo.broadcast_in_dim %447, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %449 = stablehlo.concatenate %437, %448, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %450 = mhlo.bitcast %449 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %451 = "stablehlo.gather"(%417, %450) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %452 = mhlo.bitcast %451 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %453 = stablehlo.transpose %452, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %454 = mhlo.copy %453 : tensor<16x256x128x128xf32>
      %455 = stablehlo.subtract %454, %422 : tensor<16x256x128x128xf32>
      %456 = stablehlo.convert %439 : (tensor<128xi32>) -> tensor<128xf32>
      %457 = stablehlo.subtract %428, %456 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %459 = stablehlo.clamp %423, %457, %458 : tensor<128xf32>
      %460 = stablehlo.broadcast_in_dim %459, dims = [3] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %461 = stablehlo.multiply %455, %460 : tensor<16x256x128x128xf32>
      %462 = stablehlo.add %422, %461 : tensor<16x256x128x128xf32>
      %463 = mhlo.bitcast %arg222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %464 = "stablehlo.gather"(%417, %463) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %466 = stablehlo.transpose %465, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %467 = mhlo.copy %466 : tensor<16x256x128x128xf32>
      %468 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %469 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %470 = stablehlo.add %430, %469 : tensor<128x1xi32>
      %471 = stablehlo.minimum %468, %470 : tensor<128x1xi32>
      %472 = stablehlo.compare LT, %471, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %473 = stablehlo.add %471, %433 : tensor<128x1xi32>
      %474 = stablehlo.select %472, %473, %471 : tensor<128x1xi1>, tensor<128x1xi32>
      %475 = mhlo.bitcast %474 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %477 = stablehlo.concatenate %476, %448, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %478 = mhlo.bitcast %477 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %479 = "stablehlo.gather"(%417, %478) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %480 = mhlo.bitcast %479 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %481 = stablehlo.transpose %480, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %482 = mhlo.copy %481 : tensor<16x256x128x128xf32>
      %483 = stablehlo.subtract %482, %467 : tensor<16x256x128x128xf32>
      %484 = stablehlo.multiply %483, %460 : tensor<16x256x128x128xf32>
      %485 = stablehlo.add %467, %484 : tensor<16x256x128x128xf32>
      %486 = stablehlo.subtract %485, %462 : tensor<16x256x128x128xf32>
      %487 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %488 = stablehlo.convert %430 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %489 = stablehlo.subtract %429, %488 : tensor<128x1xf32>
      %490 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %491 = stablehlo.clamp %487, %489, %490 : tensor<128x1xf32>
      %492 = mhlo.bitcast %491 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %493 = stablehlo.broadcast_in_dim %492, dims = [2] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %494 = stablehlo.multiply %486, %493 : tensor<16x256x128x128xf32>
      %495 = stablehlo.add %462, %494 : tensor<16x256x128x128xf32>
      %496 = stablehlo.broadcast_in_dim %arg220, dims = [2] : (tensor<256xf32>) -> tensor<16x1024x256xf32>
      %497 = stablehlo.add %arg219, %496 : tensor<16x1024x256xf32>
      %498 = stablehlo.transpose %497, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,1024]{1,2,0}"} : (tensor<16x1024x256xf32>) -> tensor<16x256x1024xf32>
      %499 = mhlo.copy %498 : tensor<16x256x1024xf32>
      %500 = mhlo.bitcast %499 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x1024xf32>) -> tensor<16x256x32x32xf32>
      %501 = mhlo.bitcast %arg218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %502 = "stablehlo.gather"(%500, %501) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %503 = mhlo.bitcast %502 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %504 = stablehlo.transpose %503, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %505 = mhlo.copy %504 : tensor<16x256x128x128xf32>
      %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %507 = stablehlo.multiply %arg214, %506 : tensor<128xf32>
      %508 = stablehlo.add %507, %426 : tensor<128xf32>
      %509 = stablehlo.maximum %423, %508 : tensor<128xf32>
      %510 = mhlo.bitcast %509 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %511 = stablehlo.convert %510 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %512 = stablehlo.compare LT, %511, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<32> : tensor<i32>
      %513 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %514 = stablehlo.add %511, %513 : tensor<128x1xi32>
      %515 = stablehlo.select %512, %514, %511 : tensor<128x1xi1>, tensor<128x1xi32>
      %516 = mhlo.bitcast %515 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %517 = stablehlo.broadcast_in_dim %516, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_9 = stablehlo.constant dense<31> : tensor<i32>
      %518 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %519 = stablehlo.convert %509 : (tensor<128xf32>) -> tensor<128xi32>
      %520 = stablehlo.add %519, %440 : tensor<128xi32>
      %521 = stablehlo.minimum %518, %520 : tensor<128xi32>
      %522 = stablehlo.compare LT, %521, %443 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %523 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %524 = stablehlo.add %521, %523 : tensor<128xi32>
      %525 = stablehlo.select %522, %524, %521 : tensor<128xi1>, tensor<128xi32>
      %526 = stablehlo.broadcast_in_dim %525, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %527 = stablehlo.concatenate %517, %526, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %528 = mhlo.bitcast %527 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %529 = "stablehlo.gather"(%500, %528) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %530 = mhlo.bitcast %529 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %531 = stablehlo.transpose %530, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %532 = mhlo.copy %531 : tensor<16x256x128x128xf32>
      %533 = stablehlo.subtract %532, %505 : tensor<16x256x128x128xf32>
      %534 = stablehlo.convert %519 : (tensor<128xi32>) -> tensor<128xf32>
      %535 = stablehlo.subtract %509, %534 : tensor<128xf32>
      %536 = stablehlo.clamp %423, %535, %458 : tensor<128xf32>
      %537 = stablehlo.broadcast_in_dim %536, dims = [3] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %538 = stablehlo.multiply %533, %537 : tensor<16x256x128x128xf32>
      %539 = stablehlo.add %505, %538 : tensor<16x256x128x128xf32>
      %540 = mhlo.bitcast %arg217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %541 = "stablehlo.gather"(%500, %540) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %542 = mhlo.bitcast %541 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %543 = stablehlo.transpose %542, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %544 = mhlo.copy %543 : tensor<16x256x128x128xf32>
      %545 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %546 = stablehlo.add %511, %469 : tensor<128x1xi32>
      %547 = stablehlo.minimum %545, %546 : tensor<128x1xi32>
      %548 = stablehlo.compare LT, %547, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %549 = stablehlo.add %547, %513 : tensor<128x1xi32>
      %550 = stablehlo.select %548, %549, %547 : tensor<128x1xi1>, tensor<128x1xi32>
      %551 = mhlo.bitcast %550 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %552 = stablehlo.broadcast_in_dim %551, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %553 = stablehlo.concatenate %552, %526, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %554 = mhlo.bitcast %553 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %555 = "stablehlo.gather"(%500, %554) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %556 = mhlo.bitcast %555 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %557 = stablehlo.transpose %556, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %558 = mhlo.copy %557 : tensor<16x256x128x128xf32>
      %559 = stablehlo.subtract %558, %544 : tensor<16x256x128x128xf32>
      %560 = stablehlo.multiply %559, %537 : tensor<16x256x128x128xf32>
      %561 = stablehlo.add %544, %560 : tensor<16x256x128x128xf32>
      %562 = stablehlo.subtract %561, %539 : tensor<16x256x128x128xf32>
      %563 = stablehlo.convert %511 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %564 = stablehlo.subtract %510, %563 : tensor<128x1xf32>
      %565 = stablehlo.clamp %487, %564, %490 : tensor<128x1xf32>
      %566 = mhlo.bitcast %565 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %567 = stablehlo.broadcast_in_dim %566, dims = [2] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %568 = stablehlo.multiply %562, %567 : tensor<16x256x128x128xf32>
      %569 = stablehlo.add %539, %568 : tensor<16x256x128x128xf32>
      %570 = stablehlo.broadcast_in_dim %arg216, dims = [2] : (tensor<256xf32>) -> tensor<16x4096x256xf32>
      %571 = stablehlo.add %arg215, %570 : tensor<16x4096x256xf32>
      %572 = stablehlo.transpose %571, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,4096]{1,2,0}"} : (tensor<16x4096x256xf32>) -> tensor<16x256x4096xf32>
      %573 = mhlo.copy %572 : tensor<16x256x4096xf32>
      %574 = mhlo.bitcast %573 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x4096xf32>) -> tensor<16x256x64x64xf32>
      %575 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %576 = "stablehlo.gather"(%574, %575) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %577 = mhlo.bitcast %576 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %578 = stablehlo.transpose %577, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %579 = mhlo.copy %578 : tensor<16x256x128x128xf32>
      %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %580 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %581 = stablehlo.multiply %arg214, %580 : tensor<128xf32>
      %582 = stablehlo.add %581, %426 : tensor<128xf32>
      %583 = stablehlo.maximum %423, %582 : tensor<128xf32>
      %584 = mhlo.bitcast %583 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %585 = stablehlo.convert %584 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %586 = stablehlo.compare LT, %585, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<64> : tensor<i32>
      %587 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %588 = stablehlo.add %585, %587 : tensor<128x1xi32>
      %589 = stablehlo.select %586, %588, %585 : tensor<128x1xi1>, tensor<128x1xi32>
      %590 = mhlo.bitcast %589 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %591 = stablehlo.broadcast_in_dim %590, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_12 = stablehlo.constant dense<63> : tensor<i32>
      %592 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %593 = stablehlo.convert %583 : (tensor<128xf32>) -> tensor<128xi32>
      %594 = stablehlo.add %593, %440 : tensor<128xi32>
      %595 = stablehlo.minimum %592, %594 : tensor<128xi32>
      %596 = stablehlo.compare LT, %595, %443 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %597 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %598 = stablehlo.add %595, %597 : tensor<128xi32>
      %599 = stablehlo.select %596, %598, %595 : tensor<128xi1>, tensor<128xi32>
      %600 = stablehlo.broadcast_in_dim %599, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %601 = stablehlo.concatenate %591, %600, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %602 = mhlo.bitcast %601 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %603 = "stablehlo.gather"(%574, %602) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %604 = mhlo.bitcast %603 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %605 = stablehlo.transpose %604, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %606 = mhlo.copy %605 : tensor<16x256x128x128xf32>
      %607 = stablehlo.subtract %606, %579 : tensor<16x256x128x128xf32>
      %608 = stablehlo.convert %593 : (tensor<128xi32>) -> tensor<128xf32>
      %609 = stablehlo.subtract %583, %608 : tensor<128xf32>
      %610 = stablehlo.clamp %423, %609, %458 : tensor<128xf32>
      %611 = stablehlo.broadcast_in_dim %610, dims = [3] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %612 = stablehlo.multiply %607, %611 : tensor<16x256x128x128xf32>
      %613 = stablehlo.add %579, %612 : tensor<16x256x128x128xf32>
      %614 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %615 = "stablehlo.gather"(%574, %614) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %616 = mhlo.bitcast %615 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %617 = stablehlo.transpose %616, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %618 = mhlo.copy %617 : tensor<16x256x128x128xf32>
      %619 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %620 = stablehlo.add %585, %469 : tensor<128x1xi32>
      %621 = stablehlo.minimum %619, %620 : tensor<128x1xi32>
      %622 = stablehlo.compare LT, %621, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %623 = stablehlo.add %621, %587 : tensor<128x1xi32>
      %624 = stablehlo.select %622, %623, %621 : tensor<128x1xi1>, tensor<128x1xi32>
      %625 = mhlo.bitcast %624 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %626 = stablehlo.broadcast_in_dim %625, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %627 = stablehlo.concatenate %626, %600, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %628 = mhlo.bitcast %627 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %629 = "stablehlo.gather"(%574, %628) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %630 = mhlo.bitcast %629 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %631 = stablehlo.transpose %630, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %632 = mhlo.copy %631 : tensor<16x256x128x128xf32>
      %633 = stablehlo.subtract %632, %618 : tensor<16x256x128x128xf32>
      %634 = stablehlo.multiply %633, %611 : tensor<16x256x128x128xf32>
      %635 = stablehlo.add %618, %634 : tensor<16x256x128x128xf32>
      %636 = stablehlo.subtract %635, %613 : tensor<16x256x128x128xf32>
      %637 = stablehlo.convert %585 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %638 = stablehlo.subtract %584, %637 : tensor<128x1xf32>
      %639 = stablehlo.clamp %487, %638, %490 : tensor<128x1xf32>
      %640 = mhlo.bitcast %639 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %641 = stablehlo.broadcast_in_dim %640, dims = [2] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %642 = stablehlo.multiply %636, %641 : tensor<16x256x128x128xf32>
      %643 = stablehlo.add %613, %642 : tensor<16x256x128x128xf32>
      %644 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<16x16384x256xf32>
      %645 = stablehlo.add %arg210, %644 : tensor<16x16384x256xf32>
      %646 = stablehlo.transpose %645, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,256,16384]{1,2,0}"} : (tensor<16x16384x256xf32>) -> tensor<16x256x16384xf32>
      %647 = mhlo.copy %646 : tensor<16x256x16384xf32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x256x16384xf32>) -> tensor<16x256x128x128xf32>
      %649 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %650 = "stablehlo.gather"(%648, %649) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %651 = mhlo.bitcast %650 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %652 = stablehlo.transpose %651, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %653 = mhlo.copy %652 : tensor<16x256x128x128xf32>
      %654 = stablehlo.iota dim = 0 : tensor<128xi32>
      %655 = stablehlo.convert %654 : (tensor<128xi32>) -> tensor<128xf32>
      %656 = stablehlo.maximum %423, %655 : tensor<128xf32>
      %657 = mhlo.bitcast %656 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %658 = stablehlo.convert %657 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %659 = stablehlo.compare LT, %658, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_13 = stablehlo.constant dense<128> : tensor<i32>
      %660 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %661 = stablehlo.add %658, %660 : tensor<128x1xi32>
      %662 = stablehlo.select %659, %661, %658 : tensor<128x1xi1>, tensor<128x1xi32>
      %663 = mhlo.bitcast %662 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %664 = stablehlo.broadcast_in_dim %663, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_14 = stablehlo.constant dense<127> : tensor<i32>
      %665 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %666 = stablehlo.convert %656 : (tensor<128xf32>) -> tensor<128xi32>
      %667 = stablehlo.add %666, %440 : tensor<128xi32>
      %668 = stablehlo.minimum %665, %667 : tensor<128xi32>
      %669 = stablehlo.compare LT, %668, %443 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %670 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %671 = stablehlo.add %668, %670 : tensor<128xi32>
      %672 = stablehlo.select %669, %671, %668 : tensor<128xi1>, tensor<128xi32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %674 = stablehlo.concatenate %664, %673, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %675 = mhlo.bitcast %674 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %676 = "stablehlo.gather"(%648, %675) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %678 = stablehlo.transpose %677, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %679 = mhlo.copy %678 : tensor<16x256x128x128xf32>
      %680 = stablehlo.subtract %679, %653 : tensor<16x256x128x128xf32>
      %681 = stablehlo.convert %666 : (tensor<128xi32>) -> tensor<128xf32>
      %682 = stablehlo.subtract %656, %681 : tensor<128xf32>
      %683 = stablehlo.clamp %423, %682, %458 : tensor<128xf32>
      %684 = stablehlo.broadcast_in_dim %683, dims = [3] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %685 = stablehlo.multiply %680, %684 : tensor<16x256x128x128xf32>
      %686 = stablehlo.add %653, %685 : tensor<16x256x128x128xf32>
      %687 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %688 = "stablehlo.gather"(%648, %687) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %689 = mhlo.bitcast %688 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %690 = stablehlo.transpose %689, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %691 = mhlo.copy %690 : tensor<16x256x128x128xf32>
      %692 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %693 = stablehlo.add %658, %469 : tensor<128x1xi32>
      %694 = stablehlo.minimum %692, %693 : tensor<128x1xi32>
      %695 = stablehlo.compare LT, %694, %431 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %696 = stablehlo.add %694, %660 : tensor<128x1xi32>
      %697 = stablehlo.select %695, %696, %694 : tensor<128x1xi1>, tensor<128x1xi32>
      %698 = mhlo.bitcast %697 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %699 = stablehlo.broadcast_in_dim %698, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %700 = stablehlo.concatenate %699, %673, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %701 = mhlo.bitcast %700 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %702 = "stablehlo.gather"(%648, %701) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x16x256x1x1xf32>
      %703 = mhlo.bitcast %702 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x16x256x1x1xf32>) -> tensor<128x128x16x256xf32>
      %704 = stablehlo.transpose %703, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[16,256,128,128]{1,0,3,2}"} : (tensor<128x128x16x256xf32>) -> tensor<16x256x128x128xf32>
      %705 = mhlo.copy %704 : tensor<16x256x128x128xf32>
      %706 = stablehlo.subtract %705, %691 : tensor<16x256x128x128xf32>
      %707 = stablehlo.multiply %706, %684 : tensor<16x256x128x128xf32>
      %708 = stablehlo.add %691, %707 : tensor<16x256x128x128xf32>
      %709 = stablehlo.subtract %708, %686 : tensor<16x256x128x128xf32>
      %710 = stablehlo.convert %658 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %711 = stablehlo.subtract %657, %710 : tensor<128x1xf32>
      %712 = stablehlo.clamp %487, %711, %490 : tensor<128x1xf32>
      %713 = mhlo.bitcast %712 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %714 = stablehlo.broadcast_in_dim %713, dims = [2] : (tensor<128xf32>) -> tensor<16x256x128x128xf32>
      %715 = stablehlo.multiply %709, %714 : tensor<16x256x128x128xf32>
      %716 = stablehlo.add %686, %715 : tensor<16x256x128x128xf32>
      %717 = stablehlo.concatenate %495, %569, %643, %716, dim = 1 : (tensor<16x256x128x128xf32>, tensor<16x256x128x128xf32>, tensor<16x256x128x128xf32>, tensor<16x256x128x128xf32>) -> tensor<16x1024x128x128xf32>
      %718 = stablehlo.transpose %717, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,128,128,1024]{2,1,3,0}"} : (tensor<16x1024x128x128xf32>) -> tensor<16x128x128x1024xf32>
      %719 = mhlo.copy %718 : tensor<16x128x128x1024xf32>
      mhlo.return %719 : tensor<16x128x128x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<16x16384x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<128xf32>, tensor<16x4096x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<16x1024x256xf32>, tensor<256xf32>, tensor<f32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<i32>, tensor<f32>, tensor<16x256x256xf32>, tensor<256xf32>) -> tensor<16x128x128x1024xf32>
    %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x1024xf32>) -> tensor<262144x1024xf32>
    %402 = "mhlo.fusion"(%arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x1x1xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %414 = mhlo.copy %413 : tensor<1x1x1024x256xf32>
      mhlo.return %414 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %403 = mhlo.bitcast %402 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %404 = stablehlo.dot %401, %403, precision = [DEFAULT, DEFAULT] : (tensor<262144x1024xf32>, tensor<1024x256xf32>) -> tensor<262144x256xf32>
    %405 = mhlo.bitcast %404 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<16x128x128x256xf32>
    %406 = "mhlo.fusion"(%arg203, %arg204, %0, %405, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<16x128x128x256xf32>, %arg212: tensor<256xf32>):
      %413 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<256xf32>) -> tensor<16x128x128x256xf32>
      %414 = stablehlo.subtract %arg211, %413 : tensor<16x128x128x256xf32>
      %415 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<16x128x128x256xf32>
      %416 = stablehlo.multiply %414, %415 : tensor<16x128x128x256xf32>
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<16x128x128x256xf32>
      %418 = stablehlo.multiply %416, %417 : tensor<16x128x128x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<256xf32>) -> tensor<16x128x128x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<16x128x128x256xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x128x128x256xf32>
      %422 = stablehlo.maximum %420, %421 : tensor<16x128x128x256xf32>
      mhlo.return %422 : tensor<16x128x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x128x128x256xf32>, tensor<256xf32>) -> tensor<16x128x128x256xf32>
    %407 = mhlo.bitcast %406 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x128x256xf32>) -> tensor<262144x256xf32>
    %408 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<150x256x1x1xf32>):
      %413 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,150]{1,0,2,3}"} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
      %414 = mhlo.copy %413 : tensor<1x1x256x150xf32>
      mhlo.return %414 : tensor<1x1x256x150xf32>
    }) {output_operand_aliasing = []} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
    %409 = mhlo.bitcast %408 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x150xf32>) -> tensor<256x150xf32>
    %410 = stablehlo.dot %407, %409, precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<256x150xf32>) -> tensor<262144x150xf32>
    %411 = mhlo.bitcast %410 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x150xf32>) -> tensor<16x128x128x150xf32>
    %412 = "mhlo.fusion"(%411, %arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16x128x128x150xf32>, %arg209: tensor<150xf32>):
      %413 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<150xf32>) -> tensor<16x128x128x150xf32>
      %414 = stablehlo.add %arg208, %413 : tensor<16x128x128x150xf32>
      %415 = stablehlo.transpose %414, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,150,128,128]{1,3,2,0}"} : (tensor<16x128x128x150xf32>) -> tensor<16x150x128x128xf32>
      %416 = mhlo.copy %415 : tensor<16x150x128x128xf32>
      mhlo.return %416 : tensor<16x150x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x128x150xf32>, tensor<150xf32>) -> tensor<16x150x128x128xf32>
    return %412 : tensor<16x150x128x128xf32>
  }
}
