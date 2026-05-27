module {
  func.func @main(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<384xf32>, %arg6: tensor<384xf32>, %arg7: tensor<384xf32>, %arg8: tensor<512xf32>, %arg9: tensor<512xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<256xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<128xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<64xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<32xf32>, %arg25: tensor<1x3x256x256xf32>, %arg26: tensor<32x3x3x3xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<64x32x1x1xf32>, %arg31: tensor<64xf32>, %arg32: tensor<64xf32>, %arg33: tensor<64xf32>, %arg34: tensor<64x1x3x3xf32>, %arg35: tensor<64xf32>, %arg36: tensor<64xf32>, %arg37: tensor<64xf32>, %arg38: tensor<64x64x1x1xf32>, %arg39: tensor<64xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<128x64x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x1x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128x128x1x1xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<256x128x1x1xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256x1x3x3xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<128x256x1x1xf32>, %arg63: tensor<128xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<256x128x1x1xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256x1x3x3xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256x256x1x1xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<256xf32>, %arg81: tensor<256xf32>, %arg82: tensor<128x256x1x1xf32>, %arg83: tensor<128xf32>, %arg84: tensor<128xf32>, %arg85: tensor<257x128x1x1xf32>, %arg86: tensor<257xf32>, %arg87: tensor<128x128x1x1xf32>, %arg88: tensor<128xf32>, %arg89: tensor<128xf32>, %arg90: tensor<128xf32>, %arg91: tensor<256x128x1x1xf32>, %arg92: tensor<256xf32>, %arg93: tensor<128x256x1x1xf32>, %arg94: tensor<128xf32>, %arg95: tensor<128xf32>, %arg96: tensor<128xf32>, %arg97: tensor<257x128x1x1xf32>, %arg98: tensor<257xf32>, %arg99: tensor<128x128x1x1xf32>, %arg100: tensor<128xf32>, %arg101: tensor<128xf32>, %arg102: tensor<128xf32>, %arg103: tensor<256x128x1x1xf32>, %arg104: tensor<256xf32>, %arg105: tensor<128x256x1x1xf32>, %arg106: tensor<128xf32>, %arg107: tensor<128xf32>, %arg108: tensor<128xf32>, %arg109: tensor<256x128x1x1xf32>, %arg110: tensor<256xf32>, %arg111: tensor<256xf32>, %arg112: tensor<256xf32>, %arg113: tensor<512x256x1x1xf32>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<512xf32>, %arg117: tensor<512x1x3x3xf32>, %arg118: tensor<512xf32>, %arg119: tensor<512xf32>, %arg120: tensor<512xf32>, %arg121: tensor<384x512x1x1xf32>, %arg122: tensor<384xf32>, %arg123: tensor<384xf32>, %arg124: tensor<384xf32>, %arg125: tensor<384x1x3x3xf32>, %arg126: tensor<384xf32>, %arg127: tensor<384xf32>, %arg128: tensor<384xf32>, %arg129: tensor<192x384x1x1xf32>, %arg130: tensor<192xf32>, %arg131: tensor<192xf32>, %arg132: tensor<385x192x1x1xf32>, %arg133: tensor<385xf32>, %arg134: tensor<192x192x1x1xf32>, %arg135: tensor<192xf32>, %arg136: tensor<192xf32>, %arg137: tensor<192xf32>, %arg138: tensor<384x192x1x1xf32>, %arg139: tensor<384xf32>, %arg140: tensor<192x384x1x1xf32>, %arg141: tensor<192xf32>, %arg142: tensor<192xf32>, %arg143: tensor<192xf32>, %arg144: tensor<385x192x1x1xf32>, %arg145: tensor<385xf32>, %arg146: tensor<192x192x1x1xf32>, %arg147: tensor<192xf32>, %arg148: tensor<192xf32>, %arg149: tensor<192xf32>, %arg150: tensor<384x192x1x1xf32>, %arg151: tensor<384xf32>, %arg152: tensor<192x384x1x1xf32>, %arg153: tensor<192xf32>, %arg154: tensor<192xf32>, %arg155: tensor<192xf32>, %arg156: tensor<385x192x1x1xf32>, %arg157: tensor<385xf32>, %arg158: tensor<192x192x1x1xf32>, %arg159: tensor<192xf32>, %arg160: tensor<192xf32>, %arg161: tensor<192xf32>, %arg162: tensor<384x192x1x1xf32>, %arg163: tensor<384xf32>, %arg164: tensor<192x384x1x1xf32>, %arg165: tensor<192xf32>, %arg166: tensor<192xf32>, %arg167: tensor<192xf32>, %arg168: tensor<385x192x1x1xf32>, %arg169: tensor<385xf32>, %arg170: tensor<192x192x1x1xf32>, %arg171: tensor<192xf32>, %arg172: tensor<192xf32>, %arg173: tensor<192xf32>, %arg174: tensor<384x192x1x1xf32>, %arg175: tensor<384xf32>, %arg176: tensor<192x384x1x1xf32>, %arg177: tensor<192xf32>, %arg178: tensor<192xf32>, %arg179: tensor<192xf32>, %arg180: tensor<384x192x1x1xf32>, %arg181: tensor<384xf32>, %arg182: tensor<384xf32>, %arg183: tensor<384xf32>, %arg184: tensor<768x384x1x1xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<768x1x3x3xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<768xf32>, %arg192: tensor<512x768x1x1xf32>, %arg193: tensor<512xf32>, %arg194: tensor<512xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512x1x3x3xf32>, %arg197: tensor<512xf32>, %arg198: tensor<512xf32>, %arg199: tensor<512xf32>, %arg200: tensor<256x512x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<513x256x1x1xf32>, %arg204: tensor<513xf32>, %arg205: tensor<256x256x1x1xf32>, %arg206: tensor<256xf32>, %arg207: tensor<256xf32>, %arg208: tensor<256xf32>, %arg209: tensor<512x256x1x1xf32>, %arg210: tensor<512xf32>, %arg211: tensor<256x512x1x1xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<513x256x1x1xf32>, %arg216: tensor<513xf32>, %arg217: tensor<256x256x1x1xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<512x256x1x1xf32>, %arg222: tensor<512xf32>, %arg223: tensor<256x512x1x1xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<513x256x1x1xf32>, %arg228: tensor<513xf32>, %arg229: tensor<256x256x1x1xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<512x256x1x1xf32>, %arg234: tensor<512xf32>, %arg235: tensor<256x512x1x1xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<512x256x1x1xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<2x512xf32>, %arg244: tensor<2xf32>) -> tensor<1x2xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<512xf32>
      %677 = stablehlo.rsqrt %676 : tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %675 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<1x256x8x8xf32>
      mhlo.return %675 : tensor<1x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<1x256x8x8xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %677 = stablehlo.multiply %675, %676 : tensor<4xi32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %679 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %680 = stablehlo.add %678, %679 : tensor<2x4xi32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %682 = mhlo.copy %681 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %683 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %684 = stablehlo.compare LT, %682, %683 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %685 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %686 = stablehlo.add %682, %685 : tensor<2x4x1x1xi32>
      %687 = stablehlo.select %684, %686, %682 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %688 = mhlo.bitcast %687 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %690 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %691 = stablehlo.compare LT, %680, %690 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %692 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %693 = stablehlo.add %680, %692 : tensor<2x4xi32>
      %694 = stablehlo.select %691, %693, %680 : tensor<2x4xi1>, tensor<2x4xi32>
      %695 = stablehlo.broadcast_in_dim %694, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %696 = stablehlo.concatenate %689, %695, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %697 = mhlo.bitcast %696 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      mhlo.return %697 : tensor<64x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<64x2xi32>
    %3 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<512xf32>
      %677 = stablehlo.rsqrt %676 : tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<512xf32>
      %677 = stablehlo.rsqrt %676 : tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<768xf32>
      %677 = stablehlo.rsqrt %676 : tensor<768xf32>
      mhlo.return %677 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %6 = "mhlo.fusion"(%arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<768xf32>
      %677 = stablehlo.rsqrt %676 : tensor<768xf32>
      mhlo.return %677 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %7 = "mhlo.fusion"(%arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<384xf32>
      %677 = stablehlo.rsqrt %676 : tensor<384xf32>
      mhlo.return %677 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %8 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %675 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<1x192x16x16xf32>
      mhlo.return %675 : tensor<1x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<1x192x16x16xf32>
    %9 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %677 = stablehlo.multiply %675, %676 : tensor<8xi32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %679 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %680 = stablehlo.add %678, %679 : tensor<2x8xi32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %682 = mhlo.copy %681 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %683 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %684 = stablehlo.compare LT, %682, %683 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %685 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %686 = stablehlo.add %682, %685 : tensor<2x8x1x1xi32>
      %687 = stablehlo.select %684, %686, %682 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %688 = mhlo.bitcast %687 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %690 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %691 = stablehlo.compare LT, %680, %690 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %692 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %693 = stablehlo.add %680, %692 : tensor<2x8xi32>
      %694 = stablehlo.select %691, %693, %680 : tensor<2x8xi1>, tensor<2x8xi32>
      %695 = stablehlo.broadcast_in_dim %694, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %696 = stablehlo.concatenate %689, %695, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %697 = mhlo.bitcast %696 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      mhlo.return %697 : tensor<256x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<256x2xi32>
    %10 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<384xf32>
      %677 = stablehlo.rsqrt %676 : tensor<384xf32>
      mhlo.return %677 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %11 = "mhlo.fusion"(%arg7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<384xf32>
      %677 = stablehlo.rsqrt %676 : tensor<384xf32>
      mhlo.return %677 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %12 = "mhlo.fusion"(%arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<512xf32>
      %677 = stablehlo.rsqrt %676 : tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %13 = "mhlo.fusion"(%arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<512xf32>
      %677 = stablehlo.rsqrt %676 : tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %14 = "mhlo.fusion"(%arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %675 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<1x128x32x32xf32>
      mhlo.return %675 : tensor<1x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<1x128x32x32xf32>
    %16 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %676 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %677 = stablehlo.add %675, %676 : tensor<2x16xi32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %679 = mhlo.copy %678 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %680 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %681 = stablehlo.compare LT, %679, %680 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %682 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %683 = stablehlo.add %679, %682 : tensor<2x16x1x1xi32>
      %684 = stablehlo.select %681, %683, %679 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %685 = mhlo.bitcast %684 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %686 = stablehlo.broadcast_in_dim %685, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %687 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %688 = stablehlo.compare LT, %677, %687 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %689 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %690 = stablehlo.add %677, %689 : tensor<2x16xi32>
      %691 = stablehlo.select %688, %690, %677 : tensor<2x16xi1>, tensor<2x16xi32>
      %692 = stablehlo.broadcast_in_dim %691, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %693 = stablehlo.concatenate %686, %692, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %694 = mhlo.bitcast %693 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      mhlo.return %694 : tensor<1024x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x2xi32>
    %17 = "mhlo.fusion"(%arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %20 = "mhlo.fusion"(%arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %21 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<128xf32>
      %677 = stablehlo.rsqrt %676 : tensor<128xf32>
      mhlo.return %677 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %22 = "mhlo.fusion"(%arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %23 = "mhlo.fusion"(%arg17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<256xf32>
      %677 = stablehlo.rsqrt %676 : tensor<256xf32>
      mhlo.return %677 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %24 = "mhlo.fusion"(%arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<128xf32>
      %677 = stablehlo.rsqrt %676 : tensor<128xf32>
      mhlo.return %677 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %25 = "mhlo.fusion"(%arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<128xf32>
      %677 = stablehlo.rsqrt %676 : tensor<128xf32>
      mhlo.return %677 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %26 = "mhlo.fusion"(%arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<128xf32>
      %677 = stablehlo.rsqrt %676 : tensor<128xf32>
      mhlo.return %677 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %27 = "mhlo.fusion"(%arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<64xf32>
      %677 = stablehlo.rsqrt %676 : tensor<64xf32>
      mhlo.return %677 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %28 = "mhlo.fusion"(%arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<64xf32>
      %677 = stablehlo.rsqrt %676 : tensor<64xf32>
      mhlo.return %677 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %29 = "mhlo.fusion"(%arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<64xf32>
      %677 = stablehlo.rsqrt %676 : tensor<64xf32>
      mhlo.return %677 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %30 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<32xf32>
      %677 = stablehlo.rsqrt %676 : tensor<32xf32>
      mhlo.return %677 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %31 = "mhlo.fusion"(%arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x3x256x256xf32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,256,3]{2,1,3,0}"} : (tensor<1x3x256x256xf32>) -> tensor<1x256x256x3xf32>
      %676 = mhlo.copy %675 : tensor<1x256x256x3xf32>
      mhlo.return %676 : tensor<1x256x256x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x256x256xf32>) -> tensor<1x256x256x3xf32>
    %32 = "mhlo.fusion"(%arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32x3x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,32]{1,0,2,3}"} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
      %676 = mhlo.copy %675 : tensor<3x3x3x32xf32>
      mhlo.return %676 : tensor<3x3x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
    %33 = stablehlo.convolution(%31, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x256x256x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x128x128x32xf32>
    %34 = "mhlo.fusion"(%arg27, %arg28, %30, %33, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<1x128x128x32xf32>, %arg249: tensor<32xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x128x128x32xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x128x128x32xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x128x128x32xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<32xf32>) -> tensor<1x128x128x32xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x128x128x32xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,128]{1,3,2,0}"} : (tensor<1x128x128x32xf32>) -> tensor<1x32x128x128xf32>
      %684 = mhlo.copy %683 : tensor<1x32x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x128x128xf32>
      %686 = stablehlo.negate %684 : tensor<1x32x128x128xf32>
      %687 = stablehlo.exponential %686 : tensor<1x32x128x128xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x32x128x128xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x32x128x128xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x32x128x128xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,32]{2,1,3,0}"} : (tensor<1x32x128x128xf32>) -> tensor<1x128x128x32xf32>
      %692 = mhlo.copy %691 : tensor<1x128x128x32xf32>
      mhlo.return %692 : tensor<1x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x128x128x32xf32>, tensor<32xf32>) -> tensor<1x128x128x32xf32>
    %35 = mhlo.bitcast %34 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x32xf32>) -> tensor<16384x32xf32>
    %36 = "mhlo.fusion"(%arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x32x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,32,64]{1,0,2,3}"} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
      %676 = mhlo.copy %675 : tensor<1x1x32x64xf32>
      mhlo.return %676 : tensor<1x1x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x32x64xf32>) -> tensor<32x64xf32>
    %38 = stablehlo.dot %35, %37, precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x64xf32>) -> tensor<16384x64xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<1x128x128x64xf32>
    %40 = "mhlo.fusion"(%arg31, %arg32, %29, %39, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1x128x128x64xf32>, %arg249: tensor<64xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x128x128x64xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x128x128x64xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x128x128x64xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x128x128x64xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,128,128]{1,3,2,0}"} : (tensor<1x128x128x64xf32>) -> tensor<1x64x128x128xf32>
      %684 = mhlo.copy %683 : tensor<1x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
      %686 = stablehlo.negate %684 : tensor<1x64x128x128xf32>
      %687 = stablehlo.exponential %686 : tensor<1x64x128x128xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x64x128x128xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x64x128x128xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x64x128x128xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,64]{2,1,3,0}"} : (tensor<1x64x128x128xf32>) -> tensor<1x128x128x64xf32>
      %692 = mhlo.copy %691 : tensor<1x128x128x64xf32>
      mhlo.return %692 : tensor<1x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x128x128x64xf32>, tensor<64xf32>) -> tensor<1x128x128x64xf32>
    %41 = "mhlo.fusion"(%arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,64]{1,0,2,3}"} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x64xf32>
      mhlo.return %676 : tensor<3x3x1x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x64xf32>, tensor<3x3x1x64xf32>) -> tensor<1x128x128x64xf32>
    %43 = "mhlo.fusion"(%arg35, %arg36, %28, %42, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1x128x128x64xf32>, %arg249: tensor<64xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x128x128x64xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x128x128x64xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x128x128x64xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x128x128x64xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,128,128]{1,3,2,0}"} : (tensor<1x128x128x64xf32>) -> tensor<1x64x128x128xf32>
      %684 = mhlo.copy %683 : tensor<1x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x128x128xf32>
      %686 = stablehlo.negate %684 : tensor<1x64x128x128xf32>
      %687 = stablehlo.exponential %686 : tensor<1x64x128x128xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x64x128x128xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x64x128x128xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x64x128x128xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,64]{2,1,3,0}"} : (tensor<1x64x128x128xf32>) -> tensor<1x128x128x64xf32>
      %692 = mhlo.copy %691 : tensor<1x128x128x64xf32>
      mhlo.return %692 : tensor<1x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x128x128x64xf32>, tensor<64xf32>) -> tensor<1x128x128x64xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x64xf32>) -> tensor<16384x64xf32>
    %45 = "mhlo.fusion"(%arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x64x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
      %676 = mhlo.copy %675 : tensor<1x1x64x64xf32>
      mhlo.return %676 : tensor<1x1x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
    %47 = stablehlo.dot %44, %46, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x64xf32>) -> tensor<16384x64xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<1x128x128x64xf32>
    %49 = "mhlo.fusion"(%arg39, %arg40, %27, %48, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1x128x128x64xf32>, %arg249: tensor<64xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x128x128x64xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x128x128x64xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x128x128x64xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<1x128x128x64xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x128x128x64xf32>
      mhlo.return %682 : tensor<1x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x128x128x64xf32>, tensor<64xf32>) -> tensor<1x128x128x64xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x128x64xf32>) -> tensor<16384x64xf32>
    %51 = "mhlo.fusion"(%arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x64x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x64x128xf32>
      mhlo.return %676 : tensor<1x1x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x128xf32>) -> tensor<64x128xf32>
    %53 = stablehlo.dot %50, %52, precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x128xf32>) -> tensor<16384x128xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x128xf32>) -> tensor<1x128x128x128xf32>
    %55 = "mhlo.fusion"(%arg43, %arg44, %26, %54, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1x128x128x128xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x128x128x128xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x128x128x128xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x128x128x128xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<1x128x128x128xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x128x128x128xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{1,3,2,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %684 = mhlo.copy %683 : tensor<1x128x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x128x128xf32>
      %686 = stablehlo.negate %684 : tensor<1x128x128x128xf32>
      %687 = stablehlo.exponential %686 : tensor<1x128x128x128xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x128x128x128xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x128x128x128xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x128x128x128xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,128,128]{2,1,3,0}"} : (tensor<1x128x128x128xf32>) -> tensor<1x128x128x128xf32>
      %692 = mhlo.copy %691 : tensor<1x128x128x128xf32>
      mhlo.return %692 : tensor<1x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
    %56 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x128xf32>
      mhlo.return %676 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %57 = stablehlo.convolution(%55, %56) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<1x64x64x128xf32>
    %58 = "mhlo.fusion"(%arg47, %arg48, %25, %57, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1x64x64x128xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x64x64x128xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x128xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x128xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x128xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,64,64]{1,3,2,0}"} : (tensor<1x64x64x128xf32>) -> tensor<1x128x64x64xf32>
      %684 = mhlo.copy %683 : tensor<1x128x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x64x64xf32>
      %686 = stablehlo.negate %684 : tensor<1x128x64x64xf32>
      %687 = stablehlo.exponential %686 : tensor<1x128x64x64xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x128x64x64xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x128x64x64xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x128x64x64xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,128]{2,1,3,0}"} : (tensor<1x128x64x64xf32>) -> tensor<1x64x64x128xf32>
      %692 = mhlo.copy %691 : tensor<1x64x64x128xf32>
      mhlo.return %692 : tensor<1x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x64x64x128xf32>, tensor<128xf32>) -> tensor<1x64x64x128xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x128xf32>) -> tensor<4096x128xf32>
    %60 = "mhlo.fusion"(%arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x128xf32>
      mhlo.return %676 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %62 = stablehlo.dot %59, %61, precision = [DEFAULT, DEFAULT] : (tensor<4096x128xf32>, tensor<128x128xf32>) -> tensor<4096x128xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x128xf32>) -> tensor<1x64x64x128xf32>
    %64 = "mhlo.fusion"(%arg51, %arg52, %24, %63, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1x64x64x128xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x64x64x128xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x128xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x128xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x128xf32>
      mhlo.return %682 : tensor<1x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x64x64x128xf32>, tensor<128xf32>) -> tensor<1x64x64x128xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x128xf32>) -> tensor<4096x128xf32>
    %66 = "mhlo.fusion"(%arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x256xf32>
      mhlo.return %676 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %68 = stablehlo.dot %65, %67, precision = [DEFAULT, DEFAULT] : (tensor<4096x128xf32>, tensor<128x256xf32>) -> tensor<4096x256xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<1x64x64x256xf32>
    %70 = "mhlo.fusion"(%arg55, %arg56, %23, %69, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x64x64x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x64x64x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x256xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %684 = mhlo.copy %683 : tensor<1x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %686 = stablehlo.negate %684 : tensor<1x256x64x64xf32>
      %687 = stablehlo.exponential %686 : tensor<1x256x64x64xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x256x64x64xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x256x64x64xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x256x64x64xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,256]{2,1,3,0}"} : (tensor<1x256x64x64xf32>) -> tensor<1x64x64x256xf32>
      %692 = mhlo.copy %691 : tensor<1x64x64x256xf32>
      mhlo.return %692 : tensor<1x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x64x64x256xf32>, tensor<256xf32>) -> tensor<1x64x64x256xf32>
    %71 = "mhlo.fusion"(%arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x256xf32>
      mhlo.return %676 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %72 = stablehlo.convolution(%70, %71) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<1x64x64x256xf32>
    %73 = "mhlo.fusion"(%arg59, %arg60, %22, %72, %arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x64x64x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x64x64x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x256xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %684 = mhlo.copy %683 : tensor<1x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %686 = stablehlo.negate %684 : tensor<1x256x64x64xf32>
      %687 = stablehlo.exponential %686 : tensor<1x256x64x64xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x256x64x64xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x256x64x64xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x256x64x64xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,256]{2,1,3,0}"} : (tensor<1x256x64x64xf32>) -> tensor<1x64x64x256xf32>
      %692 = mhlo.copy %691 : tensor<1x64x64x256xf32>
      mhlo.return %692 : tensor<1x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x64x64x256xf32>, tensor<256xf32>) -> tensor<1x64x64x256xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x256xf32>) -> tensor<4096x256xf32>
    %75 = "mhlo.fusion"(%arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x128xf32>
      mhlo.return %676 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %77 = stablehlo.dot %74, %76, precision = [DEFAULT, DEFAULT] : (tensor<4096x256xf32>, tensor<256x128xf32>) -> tensor<4096x128xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x128xf32>) -> tensor<1x64x64x128xf32>
    %79 = "mhlo.fusion"(%arg63, %arg64, %21, %78, %arg65, %arg51, %arg52, %24, %63, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1x64x64x128xf32>, %arg249: tensor<128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>, %arg252: tensor<128xf32>, %arg253: tensor<1x64x64x128xf32>, %arg254: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg254, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %676 = stablehlo.subtract %arg253, %675 : tensor<1x64x64x128xf32>
      %677 = stablehlo.broadcast_in_dim %arg252, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x128xf32>
      %679 = stablehlo.broadcast_in_dim %arg251, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x128xf32>
      %681 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x128xf32>
      %683 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %684 = stablehlo.subtract %arg248, %683 : tensor<1x64x64x128xf32>
      %685 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x64x64x128xf32>
      %687 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %688 = stablehlo.multiply %686, %687 : tensor<1x64x64x128xf32>
      %689 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<1x64x64x128xf32>
      %690 = stablehlo.add %688, %689 : tensor<1x64x64x128xf32>
      %691 = stablehlo.add %682, %690 : tensor<1x64x64x128xf32>
      mhlo.return %691 : tensor<1x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x64x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x64x64x128xf32>, tensor<128xf32>) -> tensor<1x64x64x128xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x64x64x128xf32>) -> tensor<4096x128xf32>
    %81 = "mhlo.fusion"(%arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x256xf32>
      mhlo.return %676 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %83 = stablehlo.dot %80, %82, precision = [DEFAULT, DEFAULT] : (tensor<4096x128xf32>, tensor<128x256xf32>) -> tensor<4096x256xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<4096x256xf32>) -> tensor<1x64x64x256xf32>
    %85 = "mhlo.fusion"(%arg67, %arg68, %20, %84, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x64x64x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x64x64x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x64x64x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x64x64x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x64x64x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x64x64x256xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,64,64]{1,3,2,0}"} : (tensor<1x64x64x256xf32>) -> tensor<1x256x64x64xf32>
      %684 = mhlo.copy %683 : tensor<1x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x64x64xf32>
      %686 = stablehlo.negate %684 : tensor<1x256x64x64xf32>
      %687 = stablehlo.exponential %686 : tensor<1x256x64x64xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x256x64x64xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x256x64x64xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x256x64x64xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,64,256]{2,1,3,0}"} : (tensor<1x256x64x64xf32>) -> tensor<1x64x64x256xf32>
      %692 = mhlo.copy %691 : tensor<1x64x64x256xf32>
      mhlo.return %692 : tensor<1x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x64x64x256xf32>, tensor<256xf32>) -> tensor<1x64x64x256xf32>
    %86 = "mhlo.fusion"(%arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x256xf32>
      mhlo.return %676 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %87 = stablehlo.convolution(%85, %86) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<1x32x32x256xf32>
    %88 = "mhlo.fusion"(%arg71, %arg72, %19, %87, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x32x32x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x32x32x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x32x32x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x32x32x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x32x32x256xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,32,32]{1,3,2,0}"} : (tensor<1x32x32x256xf32>) -> tensor<1x256x32x32xf32>
      %684 = mhlo.copy %683 : tensor<1x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
      %686 = stablehlo.negate %684 : tensor<1x256x32x32xf32>
      %687 = stablehlo.exponential %686 : tensor<1x256x32x32xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x256x32x32xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x256x32x32xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x256x32x32xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,256]{2,1,3,0}"} : (tensor<1x256x32x32xf32>) -> tensor<1x32x32x256xf32>
      %692 = mhlo.copy %691 : tensor<1x32x32x256xf32>
      mhlo.return %692 : tensor<1x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x32x32x256xf32>, tensor<256xf32>) -> tensor<1x32x32x256xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x32x256xf32>) -> tensor<1024x256xf32>
    %90 = "mhlo.fusion"(%arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x256xf32>
      mhlo.return %676 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %92 = stablehlo.dot %89, %91, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x32x32x256xf32>
    %94 = "mhlo.fusion"(%arg75, %arg76, %18, %93, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x32x32x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x32x32x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x32x32x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x32x32x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x32x32x256xf32>
      mhlo.return %682 : tensor<1x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x32x32x256xf32>, tensor<256xf32>) -> tensor<1x32x32x256xf32>
    %95 = "mhlo.fusion"(%arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x256xf32>
      mhlo.return %676 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %96 = stablehlo.convolution(%94, %95) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x256xf32>, tensor<3x3x1x256xf32>) -> tensor<1x32x32x256xf32>
    %97 = "mhlo.fusion"(%arg79, %arg80, %17, %96, %arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x32x32x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x32x32x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x32x32x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x32x32x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x32x32x256xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,32,32]{1,3,2,0}"} : (tensor<1x32x32x256xf32>) -> tensor<1x256x32x32xf32>
      %684 = mhlo.copy %683 : tensor<1x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x32x32xf32>
      %686 = stablehlo.negate %684 : tensor<1x256x32x32xf32>
      %687 = stablehlo.exponential %686 : tensor<1x256x32x32xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x256x32x32xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x256x32x32xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x256x32x32xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,256]{2,1,3,0}"} : (tensor<1x256x32x32xf32>) -> tensor<1x32x32x256xf32>
      %692 = mhlo.copy %691 : tensor<1x32x32x256xf32>
      mhlo.return %692 : tensor<1x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x32x32x256xf32>, tensor<256xf32>) -> tensor<1x32x32x256xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x32x256xf32>) -> tensor<1024x256xf32>
    %99 = "mhlo.fusion"(%arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x128xf32>
      mhlo.return %676 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %100 = mhlo.bitcast %99 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %101 = stablehlo.dot %98, %100, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x128xf32>) -> tensor<1x32x32x128xf32>
    %103 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %676 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %677 = stablehlo.add %675, %676 : tensor<2x16xi32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %679 = mhlo.copy %678 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %680 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %681 = stablehlo.compare LT, %679, %680 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %682 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %683 = stablehlo.add %679, %682 : tensor<2x16x1x1xi32>
      %684 = stablehlo.select %681, %683, %679 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %685 = mhlo.bitcast %684 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %686 = stablehlo.broadcast_in_dim %685, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %687 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %688 = stablehlo.compare LT, %677, %687 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %689 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %690 = stablehlo.add %677, %689 : tensor<2x16xi32>
      %691 = stablehlo.select %688, %690, %677 : tensor<2x16xi1>, tensor<2x16xi32>
      %692 = stablehlo.broadcast_in_dim %691, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %693 = stablehlo.concatenate %686, %692, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      mhlo.return %693 : tensor<2x16x2x16x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x16x2x16x2xi32>
    %104 = "mhlo.fusion"(%102, %103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x32x32x128xf32>, %arg246: tensor<2x16x2x16x2xi32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,32,32]{1,3,2,0}"} : (tensor<1x32x32x128xf32>) -> tensor<1x128x32x32xf32>
      %676 = mhlo.copy %675 : tensor<1x128x32x32xf32>
      %677 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %678 = "stablehlo.gather"(%676, %677) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128, 1, 1>}> : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1x128x1x1xf32>
      mhlo.return %678 : tensor<1024x1x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x32x32x128xf32>, tensor<2x16x2x16x2xi32>) -> tensor<1024x1x128x1x1xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x1x128x1x1xf32>):
      %675 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<2x16x2x16x1x128xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x1x128xf32>) -> tensor<1x128x2x2x16x16xf32>
      %677 = mhlo.copy %676 : tensor<1x128x2x2x16x16xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x1x131072xf32>
      mhlo.return %678 : tensor<1x1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x128x1x1xf32>) -> tensor<1x1x131072xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
      mhlo.return %675 : tensor<1x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %109 = "mhlo.fusion"(%108, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1024x1x128x1x1xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<2x16x2x16x1x128xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x1x128xf32>) -> tensor<1x128x2x2x16x16xf32>
      %677 = mhlo.copy %676 : tensor<1x128x2x2x16x16xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %679 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %682 = stablehlo.subtract %678, %681 : tensor<1x1x131072xf32>
      %683 = stablehlo.multiply %682, %682 : tensor<1x1x131072xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x131072xf32>
      mhlo.return %684 : tensor<1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x131072xf32>
    %110 = "mhlo.fusion"(%109, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %675 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %111 = "mhlo.fusion"(%110, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4096xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
      mhlo.return %675 : tensor<1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
    %112 = "mhlo.fusion"(%111, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
      mhlo.return %675 : tensor<1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4xf32>) -> tensor<1xf32>
    %114 = "mhlo.fusion"(%arg83, %113, %108, %104, %arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1024x1x128x1x1xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<2x16x2x16x1x128xf32>
      %677 = stablehlo.transpose %676, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x1x128xf32>) -> tensor<1x128x2x2x16x16xf32>
      %678 = mhlo.copy %677 : tensor<1x128x2x2x16x16xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %680 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %683 = stablehlo.subtract %679, %682 : tensor<1x1x131072xf32>
      %684 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %685 = stablehlo.broadcast_in_dim %684, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %686 = stablehlo.multiply %683, %685 : tensor<1x1x131072xf32>
      %687 = mhlo.bitcast %686 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
      %688 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %689 = stablehlo.multiply %687, %688 : tensor<1x128x4x256xf32>
      %690 = stablehlo.add %675, %689 : tensor<1x128x4x256xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %692 = mhlo.copy %691 : tensor<1x4x256x128xf32>
      mhlo.return %692 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1024x1x128x1x1xf32>, tensor<128xf32>) -> tensor<1x4x256x128xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %116 = "mhlo.fusion"(%arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x257xf32>
      mhlo.return %676 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %118 = stablehlo.dot %115, %117, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x257xf32>) -> tensor<1024x257xf32>
    %119 = mhlo.bitcast %118 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x257xf32>) -> tensor<1x4x256x257xf32>
    %120 = "mhlo.fusion"(%119, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x256x257xf32>, %arg246: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      mhlo.return %679 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x1x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %121 = "mhlo.fusion"(%120, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
      mhlo.return %675 : tensor<1x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
    %122 = "mhlo.fusion"(%121, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %123 = "mhlo.fusion"(%122, %119, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x256xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x256xf32>
      mhlo.return %685 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x1x4x256xf32>
    %124 = "mhlo.fusion"(%123, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
      mhlo.return %675 : tensor<1x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %126 = "mhlo.fusion"(%123, %125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x256xf32>
      mhlo.return %677 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x256xf32>
    %127 = "mhlo.fusion"(%126, %119, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x128x4x256xf32>
      mhlo.return %682 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x128x4x256xf32>
    %128 = "mhlo.fusion"(%127, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<1x128x4x8xf32>
      mhlo.return %675 : tensor<1x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<1x128x4x8xf32>
    %129 = "mhlo.fusion"(%128, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x128x4x8xf32>, tensor<f32>) -> tensor<1x128x4xf32>
      mhlo.return %675 : tensor<1x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x8xf32>, tensor<f32>) -> tensor<1x128x4xf32>
    %130 = "mhlo.fusion"(%129, %119, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x128x4x256xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x4xf32>) -> tensor<128x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x128x4x256xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %686 = mhlo.copy %685 : tensor<1x4x256x128xf32>
      mhlo.return %686 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x4x256x128xf32>
    %131 = mhlo.bitcast %130 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %132 = "mhlo.fusion"(%arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x128xf32>
      mhlo.return %676 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %133 = mhlo.bitcast %132 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %134 = stablehlo.dot %131, %133, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x128xf32>) -> tensor<1024x128xf32>
    %135 = mhlo.bitcast %134 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x128xf32>) -> tensor<1x4x256x128xf32>
    %136 = "mhlo.fusion"(%135, %arg88, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x256x128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1024x1x128x1x1xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x256x128xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x128x4x256xf32>
      %679 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1x128x1x1xf32>) -> tensor<2x16x2x16x1x128xf32>
      %680 = stablehlo.transpose %679, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x1x128xf32>) -> tensor<1x128x2x2x16x16xf32>
      %681 = mhlo.copy %680 : tensor<1x128x2x2x16x16xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x128x2x2x16x16xf32>) -> tensor<1x128x4x256xf32>
      %683 = stablehlo.add %678, %682 : tensor<1x128x4x256xf32>
      mhlo.return %683 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x256x128xf32>, tensor<128xf32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x128x4x256xf32>
    %137 = mhlo.bitcast %136 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
    %138 = "mhlo.fusion"(%137, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
      mhlo.return %675 : tensor<1x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %139 = "mhlo.fusion"(%138, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %140 = "mhlo.fusion"(%139, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %141 = "mhlo.fusion"(%140, %136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x128x4x256xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x131072xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x131072xf32>
      mhlo.return %681 : tensor<1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x128x4x256xf32>) -> tensor<1x131072xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %675 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %143 = "mhlo.fusion"(%142, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4096xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
      mhlo.return %675 : tensor<1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
    %144 = "mhlo.fusion"(%143, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
      mhlo.return %675 : tensor<1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4xf32>) -> tensor<1xf32>
    %146 = "mhlo.fusion"(%arg89, %145, %140, %136, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x128x4x256xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x131072xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x128x4x256xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x128x4x256xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %689 = mhlo.copy %688 : tensor<1x4x256x128xf32>
      mhlo.return %689 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x128x4x256xf32>, tensor<128xf32>) -> tensor<1x4x256x128xf32>
    %147 = mhlo.bitcast %146 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %148 = "mhlo.fusion"(%arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x256xf32>
      mhlo.return %676 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %150 = stablehlo.dot %147, %149, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x4x256x256xf32>
    %152 = "mhlo.fusion"(%151, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x256x256xf32>, %arg246: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x4x256x256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x256x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,256]{1,3,2,0}"} : (tensor<1x4x256x256xf32>) -> tensor<1x256x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
      %680 = stablehlo.negate %678 : tensor<1x256x4x256xf32>
      %681 = stablehlo.exponential %680 : tensor<1x256x4x256xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x256x4x256xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x256x4x256xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x256x4x256xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,256]{2,1,3,0}"} : (tensor<1x256x4x256xf32>) -> tensor<1x4x256x256xf32>
      %686 = mhlo.copy %685 : tensor<1x4x256x256xf32>
      mhlo.return %686 : tensor<1x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x256x256xf32>, tensor<256xf32>) -> tensor<1x4x256x256xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x256xf32>) -> tensor<1024x256xf32>
    %154 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x128xf32>
      mhlo.return %676 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %155 = mhlo.bitcast %154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %156 = stablehlo.dot %153, %155, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x128xf32>) -> tensor<1x4x256x128xf32>
    %158 = "mhlo.fusion"(%136, %157, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<1x4x256x128xf32>, %arg247: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x128xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x128x4x256xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x128x4x256xf32>
      mhlo.return %679 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<1x4x256x128xf32>, tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %159 = mhlo.bitcast %158 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
    %160 = "mhlo.fusion"(%159, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
      mhlo.return %675 : tensor<1x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %161 = "mhlo.fusion"(%160, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %162 = "mhlo.fusion"(%161, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %163 = "mhlo.fusion"(%162, %158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x128x4x256xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x131072xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x131072xf32>
      mhlo.return %681 : tensor<1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x128x4x256xf32>) -> tensor<1x131072xf32>
    %164 = "mhlo.fusion"(%163, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %675 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %165 = "mhlo.fusion"(%164, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4096xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
      mhlo.return %675 : tensor<1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
    %166 = "mhlo.fusion"(%165, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
      mhlo.return %675 : tensor<1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
    %167 = "mhlo.fusion"(%166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4xf32>) -> tensor<1xf32>
    %168 = "mhlo.fusion"(%arg95, %167, %162, %158, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x128x4x256xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x131072xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x128x4x256xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x128x4x256xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %689 = mhlo.copy %688 : tensor<1x4x256x128xf32>
      mhlo.return %689 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x128x4x256xf32>, tensor<128xf32>) -> tensor<1x4x256x128xf32>
    %169 = mhlo.bitcast %168 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %170 = "mhlo.fusion"(%arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x257xf32>
      mhlo.return %676 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %172 = stablehlo.dot %169, %171, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x257xf32>) -> tensor<1024x257xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x257xf32>) -> tensor<1x4x256x257xf32>
    %174 = "mhlo.fusion"(%173, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x256x257xf32>, %arg246: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      mhlo.return %679 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x1x4x256xf32>
    %175 = "mhlo.fusion"(%174, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
      mhlo.return %675 : tensor<1x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
    %176 = "mhlo.fusion"(%175, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %177 = "mhlo.fusion"(%176, %173, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x256xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x256xf32>
      mhlo.return %685 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x1x4x256xf32>
    %178 = "mhlo.fusion"(%177, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
      mhlo.return %675 : tensor<1x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<f32>) -> tensor<1x1x4x8xf32>
    %179 = "mhlo.fusion"(%178, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %180 = "mhlo.fusion"(%177, %179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x256xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x256xf32>
      mhlo.return %677 : tensor<1x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x256xf32>
    %181 = "mhlo.fusion"(%180, %173, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x256xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 1:129, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x256xf32>) -> tensor<4x256xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x256xf32>) -> tensor<1x128x4x256xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x128x4x256xf32>
      mhlo.return %682 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x256xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x128x4x256xf32>
    %182 = "mhlo.fusion"(%181, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<1x128x4x8xf32>
      mhlo.return %675 : tensor<1x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<f32>) -> tensor<1x128x4x8xf32>
    %183 = "mhlo.fusion"(%182, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x8xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x128x4x8xf32>, tensor<f32>) -> tensor<1x128x4xf32>
      mhlo.return %675 : tensor<1x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x8xf32>, tensor<f32>) -> tensor<1x128x4xf32>
    %184 = "mhlo.fusion"(%183, %173, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4xf32>, %arg246: tensor<1x4x256x257xf32>, %arg247: tensor<257xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<1x4x256x257xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x257xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,257,4,256]{1,3,2,0}"} : (tensor<1x4x256x257xf32>) -> tensor<1x257x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x257x4x256xf32>
      %679 = stablehlo.slice %678 [0:1, 129:257, 0:4, 0:256] : (tensor<1x257x4x256xf32>) -> tensor<1x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x4x256xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x128x4x256xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x128x4xf32>) -> tensor<128x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<128x4xf32>) -> tensor<1x128x4x256xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x128x4x256xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %686 = mhlo.copy %685 : tensor<1x4x256x128xf32>
      mhlo.return %686 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4xf32>, tensor<1x4x256x257xf32>, tensor<257xf32>) -> tensor<1x4x256x128xf32>
    %185 = mhlo.bitcast %184 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %186 = "mhlo.fusion"(%arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x128xf32>
      mhlo.return %676 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %187 = mhlo.bitcast %186 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %188 = stablehlo.dot %185, %187, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x128xf32>) -> tensor<1024x128xf32>
    %189 = mhlo.bitcast %188 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x128xf32>) -> tensor<1x4x256x128xf32>
    %190 = "mhlo.fusion"(%158, %189, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<1x4x256x128xf32>, %arg247: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x128xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x128x4x256xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x128x4x256xf32>
      mhlo.return %679 : tensor<1x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<1x4x256x128xf32>, tensor<128xf32>) -> tensor<1x128x4x256xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
    %192 = "mhlo.fusion"(%191, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
      mhlo.return %675 : tensor<1x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %193 = "mhlo.fusion"(%192, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %194 = "mhlo.fusion"(%193, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %195 = "mhlo.fusion"(%194, %190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x128x4x256xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x131072xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x131072xf32>
      mhlo.return %681 : tensor<1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x128x4x256xf32>) -> tensor<1x131072xf32>
    %196 = "mhlo.fusion"(%195, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %675 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %197 = "mhlo.fusion"(%196, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4096xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
      mhlo.return %675 : tensor<1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
    %198 = "mhlo.fusion"(%197, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
      mhlo.return %675 : tensor<1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4xf32>) -> tensor<1xf32>
    %200 = "mhlo.fusion"(%arg101, %199, %194, %190, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x128x4x256xf32>, %arg249: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x131072xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x131072xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x128x4x256xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x128x4x256xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,128]{2,1,3,0}"} : (tensor<1x128x4x256xf32>) -> tensor<1x4x256x128xf32>
      %689 = mhlo.copy %688 : tensor<1x4x256x128xf32>
      mhlo.return %689 : tensor<1x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x128x4x256xf32>, tensor<128xf32>) -> tensor<1x4x256x128xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x128xf32>) -> tensor<1024x128xf32>
    %202 = "mhlo.fusion"(%arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x256xf32>
      mhlo.return %676 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %204 = stablehlo.dot %201, %203, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
    %205 = mhlo.bitcast %204 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x4x256x256xf32>
    %206 = "mhlo.fusion"(%205, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x256x256xf32>, %arg246: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x4x256x256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x256x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,256]{1,3,2,0}"} : (tensor<1x4x256x256xf32>) -> tensor<1x256x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4x256xf32>
      %680 = stablehlo.negate %678 : tensor<1x256x4x256xf32>
      %681 = stablehlo.exponential %680 : tensor<1x256x4x256xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x256x4x256xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x256x4x256xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x256x4x256xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,256,256]{2,1,3,0}"} : (tensor<1x256x4x256xf32>) -> tensor<1x4x256x256xf32>
      %686 = mhlo.copy %685 : tensor<1x4x256x256xf32>
      mhlo.return %686 : tensor<1x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x256x256xf32>, tensor<256xf32>) -> tensor<1x4x256x256xf32>
    %207 = mhlo.bitcast %206 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x256x256xf32>) -> tensor<1024x256xf32>
    %208 = "mhlo.fusion"(%arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x128xf32>
      mhlo.return %676 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %209 = mhlo.bitcast %208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %210 = stablehlo.dot %207, %209, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x128xf32>) -> tensor<1024x128xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x128xf32>) -> tensor<1x4x256x128xf32>
    %212 = "mhlo.fusion"(%190, %211, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x4x256xf32>, %arg246: tensor<1x4x256x128xf32>, %arg247: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x256x128xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x128x4x256xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x128x4x256xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      mhlo.return %680 : tensor<1x1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x4x256xf32>, tensor<1x4x256x128xf32>, tensor<128xf32>) -> tensor<1x1x131072xf32>
    %213 = "mhlo.fusion"(%212, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
      mhlo.return %675 : tensor<1x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x131072xf32>, tensor<f32>) -> tensor<1x1x128xf32>
    %214 = "mhlo.fusion"(%213, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %215 = "mhlo.fusion"(%214, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<f32>) -> tensor<1xf32>
    %216 = "mhlo.fusion"(%215, %190, %211, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x128x4x256xf32>, %arg247: tensor<1x4x256x128xf32>, %arg248: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %676 = stablehlo.add %arg247, %675 : tensor<1x4x256x128xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %678 = mhlo.copy %677 : tensor<1x128x4x256xf32>
      %679 = stablehlo.add %678, %arg246 : tensor<1x128x4x256xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %681 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %684 = stablehlo.subtract %680, %683 : tensor<1x1x131072xf32>
      %685 = stablehlo.multiply %684, %684 : tensor<1x1x131072xf32>
      %686 = mhlo.bitcast %685 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x131072xf32>
      mhlo.return %686 : tensor<1x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x128x4x256xf32>, tensor<1x4x256x128xf32>, tensor<128xf32>) -> tensor<1x131072xf32>
    %217 = "mhlo.fusion"(%216, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x131072xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
      mhlo.return %675 : tensor<1x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<1x131072xf32>, tensor<f32>) -> tensor<1x4096xf32>
    %218 = "mhlo.fusion"(%217, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4096xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
      mhlo.return %675 : tensor<1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1x128xf32>
    %219 = "mhlo.fusion"(%218, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
      mhlo.return %675 : tensor<1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x4xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4xf32>) -> tensor<1xf32>
    %221 = "mhlo.fusion"(%arg107, %220, %215, %190, %211, %arg106, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x128x4x256xf32>, %arg249: tensor<1x4x256x128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>):
      %675 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %676 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<1x4x256x128xf32>
      %677 = stablehlo.add %arg249, %676 : tensor<1x4x256x128xf32>
      %678 = stablehlo.transpose %677, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,256]{1,3,2,0}"} : (tensor<1x4x256x128xf32>) -> tensor<1x128x4x256xf32>
      %679 = mhlo.copy %678 : tensor<1x128x4x256xf32>
      %680 = stablehlo.add %679, %arg248 : tensor<1x128x4x256xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x1x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<1xf32>
      %682 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %683 = mhlo.bitcast %682 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %684 = stablehlo.broadcast_in_dim %683, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %685 = stablehlo.subtract %681, %684 : tensor<1x1x131072xf32>
      %686 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %687 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<f32>) -> tensor<1x1x131072xf32>
      %688 = stablehlo.multiply %685, %687 : tensor<1x1x131072xf32>
      %689 = mhlo.bitcast %688 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x131072xf32>) -> tensor<1x128x4x256xf32>
      %690 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1x128x4x256xf32>
      %691 = stablehlo.multiply %689, %690 : tensor<1x128x4x256xf32>
      %692 = stablehlo.add %675, %691 : tensor<1x128x4x256xf32>
      %693 = mhlo.bitcast %692 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x4x256xf32>) -> tensor<1x128x2x2x16x16xf32>
      %694 = stablehlo.transpose %693, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,16,2,16,1,128]{3,1,2,0,5,4}"} : (tensor<1x128x2x2x16x16xf32>) -> tensor<2x16x2x16x1x128xf32>
      %695 = mhlo.copy %694 : tensor<2x16x2x16x1x128xf32>
      %696 = mhlo.bitcast %695 {result_layout = dense<[2, 0, 4, 3, 1]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1024,1,128,1,1]{2,0,4,3,1}"} : (tensor<2x16x2x16x1x128xf32>) -> tensor<1024x1x128x1x1xf32>
      %697 = mhlo.copy %696 : tensor<1024x1x128x1x1xf32>
      mhlo.return %697 : tensor<1024x1x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x128x4x256xf32>, tensor<1x4x256x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1024x1x128x1x1xf32>
    %222 = "mhlo.fusion"(%15, %16, %221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x32x32xf32>, %arg246: tensor<1024x2xi32>, %arg247: tensor<1024x1x128x1x1xf32>):
      %675 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %676 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x128x32x32xf32>
      mhlo.return %675 : tensor<1x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1x128x1x1xf32>) -> tensor<1x128x32x32xf32>
    %223 = "mhlo.fusion"(%222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x128x32x32xf32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,128]{2,1,3,0}"} : (tensor<1x128x32x32xf32>) -> tensor<1x32x32x128xf32>
      %676 = mhlo.copy %675 : tensor<1x32x32x128xf32>
      mhlo.return %676 : tensor<1x32x32x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x128x32x32xf32>) -> tensor<1x32x32x128xf32>
    %224 = mhlo.bitcast %223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x32x128xf32>) -> tensor<1024x128xf32>
    %225 = "mhlo.fusion"(%arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x128x256xf32>
      mhlo.return %676 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %227 = stablehlo.dot %224, %226, precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<128x256xf32>) -> tensor<1024x256xf32>
    %228 = mhlo.bitcast %227 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x256xf32>) -> tensor<1x32x32x256xf32>
    %229 = "mhlo.fusion"(%arg110, %arg111, %14, %228, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1x32x32x256xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x32x32x256xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x32x32x256xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x32x32x256xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<1x32x32x256xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x32x32x256xf32>
      mhlo.return %682 : tensor<1x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x32x32x256xf32>, tensor<256xf32>) -> tensor<1x32x32x256xf32>
    %230 = mhlo.bitcast %229 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x32x32x256xf32>) -> tensor<1024x256xf32>
    %231 = "mhlo.fusion"(%arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x512xf32>
      mhlo.return %676 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %232 = mhlo.bitcast %231 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %233 = stablehlo.dot %230, %232, precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x512xf32>) -> tensor<1024x512xf32>
    %234 = mhlo.bitcast %233 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1024x512xf32>) -> tensor<1x32x32x512xf32>
    %235 = "mhlo.fusion"(%arg114, %arg115, %13, %234, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1x32x32x512xf32>, %arg249: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x32x32x512xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x32x32x512xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x32x32x512xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x32x32x512xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x32x32x512xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x32x32x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x32x32x512xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x32x32x512xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,32,32]{1,3,2,0}"} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
      %684 = mhlo.copy %683 : tensor<1x512x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x32x32xf32>
      %686 = stablehlo.negate %684 : tensor<1x512x32x32xf32>
      %687 = stablehlo.exponential %686 : tensor<1x512x32x32xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x512x32x32xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x512x32x32xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x512x32x32xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,512]{2,1,3,0}"} : (tensor<1x512x32x32xf32>) -> tensor<1x32x32x512xf32>
      %692 = mhlo.copy %691 : tensor<1x32x32x512xf32>
      mhlo.return %692 : tensor<1x32x32x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x32x32x512xf32>, tensor<512xf32>) -> tensor<1x32x32x512xf32>
    %236 = "mhlo.fusion"(%arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x512xf32>
      mhlo.return %676 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %237 = stablehlo.convolution(%235, %236) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x512xf32>, tensor<3x3x1x512xf32>) -> tensor<1x16x16x512xf32>
    %238 = "mhlo.fusion"(%arg118, %arg119, %12, %237, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1x16x16x512xf32>, %arg249: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x16x16x512xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x16x16x512xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x16x16x512xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x16x16x512xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x16x16x512xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x16x16x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x16x16x512xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x16x16x512xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,16,16]{1,3,2,0}"} : (tensor<1x16x16x512xf32>) -> tensor<1x512x16x16xf32>
      %684 = mhlo.copy %683 : tensor<1x512x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x16x16xf32>
      %686 = stablehlo.negate %684 : tensor<1x512x16x16xf32>
      %687 = stablehlo.exponential %686 : tensor<1x512x16x16xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x512x16x16xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x512x16x16xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x512x16x16xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,512]{2,1,3,0}"} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
      %692 = mhlo.copy %691 : tensor<1x16x16x512xf32>
      mhlo.return %692 : tensor<1x16x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x16x16x512xf32>, tensor<512xf32>) -> tensor<1x16x16x512xf32>
    %239 = mhlo.bitcast %238 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x512xf32>) -> tensor<256x512xf32>
    %240 = "mhlo.fusion"(%arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x512x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,384]{1,0,2,3}"} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x512x384xf32>
      mhlo.return %676 : tensor<1x1x512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
    %241 = mhlo.bitcast %240 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x384xf32>) -> tensor<512x384xf32>
    %242 = stablehlo.dot %239, %241, precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x384xf32>) -> tensor<256x384xf32>
    %243 = mhlo.bitcast %242 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x16x16x384xf32>
    %244 = "mhlo.fusion"(%arg122, %arg123, %11, %243, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1x16x16x384xf32>, %arg249: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x16x16x384xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x16x16x384xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x16x16x384xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x16x16x384xf32>
      mhlo.return %682 : tensor<1x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1x16x16x384xf32>, tensor<384xf32>) -> tensor<1x16x16x384xf32>
    %245 = "mhlo.fusion"(%arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,384]{1,0,2,3}"} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x384xf32>
      mhlo.return %676 : tensor<3x3x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
    %246 = stablehlo.convolution(%244, %245) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x16x16x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x16x16x384xf32>
    %247 = "mhlo.fusion"(%arg126, %arg127, %10, %246, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1x16x16x384xf32>, %arg249: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x16x16x384xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x16x16x384xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x16x16x384xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x16x16x384xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,384,16,16]{1,3,2,0}"} : (tensor<1x16x16x384xf32>) -> tensor<1x384x16x16xf32>
      %684 = mhlo.copy %683 : tensor<1x384x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x384x16x16xf32>
      %686 = stablehlo.negate %684 : tensor<1x384x16x16xf32>
      %687 = stablehlo.exponential %686 : tensor<1x384x16x16xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x384x16x16xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x384x16x16xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x384x16x16xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,384]{2,1,3,0}"} : (tensor<1x384x16x16xf32>) -> tensor<1x16x16x384xf32>
      %692 = mhlo.copy %691 : tensor<1x16x16x384xf32>
      mhlo.return %692 : tensor<1x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1x16x16x384xf32>, tensor<384xf32>) -> tensor<1x16x16x384xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x384xf32>) -> tensor<256x384xf32>
    %249 = "mhlo.fusion"(%arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x192xf32>
      mhlo.return %676 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %250 = mhlo.bitcast %249 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %251 = stablehlo.dot %248, %250, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x192xf32>) -> tensor<256x192xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x16x16x192xf32>
    %253 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %677 = stablehlo.multiply %675, %676 : tensor<8xi32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %679 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %680 = stablehlo.add %678, %679 : tensor<2x8xi32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %682 = mhlo.copy %681 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %683 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %684 = stablehlo.compare LT, %682, %683 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %685 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %686 = stablehlo.add %682, %685 : tensor<2x8x1x1xi32>
      %687 = stablehlo.select %684, %686, %682 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %688 = mhlo.bitcast %687 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %690 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %691 = stablehlo.compare LT, %680, %690 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %692 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %693 = stablehlo.add %680, %692 : tensor<2x8xi32>
      %694 = stablehlo.select %691, %693, %680 : tensor<2x8xi1>, tensor<2x8xi32>
      %695 = stablehlo.broadcast_in_dim %694, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %696 = stablehlo.concatenate %689, %695, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      mhlo.return %696 : tensor<2x8x2x8x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x8x2x8x2xi32>
    %254 = "mhlo.fusion"(%252, %253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16x16x192xf32>, %arg246: tensor<2x8x2x8x2xi32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,16,16]{1,3,2,0}"} : (tensor<1x16x16x192xf32>) -> tensor<1x192x16x16xf32>
      %676 = mhlo.copy %675 : tensor<1x192x16x16xf32>
      %677 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %678 = "stablehlo.gather"(%676, %677) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 192, 1, 1>}> : (tensor<1x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1x192x1x1xf32>
      mhlo.return %678 : tensor<256x1x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x192xf32>, tensor<2x8x2x8x2xi32>) -> tensor<256x1x192x1x1xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x192x1x1xf32>):
      %675 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<2x8x2x8x1x192xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x1x192xf32>) -> tensor<1x192x2x2x8x8xf32>
      %677 = mhlo.copy %676 : tensor<1x192x2x2x8x8xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x1x49152xf32>
      mhlo.return %678 : tensor<1x1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x192x1x1xf32>) -> tensor<1x1x49152xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %257 = "mhlo.fusion"(%256, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %258 = "mhlo.fusion"(%257, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %259 = "mhlo.fusion"(%258, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<256x1x192x1x1xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<2x8x2x8x1x192xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x1x192xf32>) -> tensor<1x192x2x2x8x8xf32>
      %677 = mhlo.copy %676 : tensor<1x192x2x2x8x8xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %679 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %682 = stablehlo.subtract %678, %681 : tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %682, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %684 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<256x1x192x1x1xf32>) -> tensor<1x49152xf32>
    %260 = "mhlo.fusion"(%259, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %261 = "mhlo.fusion"(%260, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %262 = "mhlo.fusion"(%261, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %263 = "mhlo.fusion"(%262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %264 = "mhlo.fusion"(%arg130, %263, %258, %254, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<256x1x192x1x1xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<2x8x2x8x1x192xf32>
      %677 = stablehlo.transpose %676, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x1x192xf32>) -> tensor<1x192x2x2x8x8xf32>
      %678 = mhlo.copy %677 : tensor<1x192x2x2x8x8xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %680 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.subtract %679, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %685 = stablehlo.broadcast_in_dim %684, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %686 = stablehlo.multiply %683, %685 : tensor<1x1x49152xf32>
      %687 = mhlo.bitcast %686 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %688 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %689 = stablehlo.multiply %687, %688 : tensor<1x192x4x64xf32>
      %690 = stablehlo.add %675, %689 : tensor<1x192x4x64xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %692 = mhlo.copy %691 : tensor<1x4x64x192xf32>
      mhlo.return %692 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<256x1x192x1x1xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %265 = mhlo.bitcast %264 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %266 = "mhlo.fusion"(%arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x385xf32>
      mhlo.return %676 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %268 = stablehlo.dot %265, %267, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x385xf32>) -> tensor<256x385xf32>
    %269 = mhlo.bitcast %268 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x385xf32>) -> tensor<1x4x64x385xf32>
    %270 = "mhlo.fusion"(%269, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x385xf32>, %arg246: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      mhlo.return %679 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %271 = "mhlo.fusion"(%270, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %272 = "mhlo.fusion"(%271, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %273 = "mhlo.fusion"(%272, %269, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x64xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x64xf32>
      mhlo.return %685 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %275 = "mhlo.fusion"(%274, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %276 = "mhlo.fusion"(%273, %275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x64xf32>
      mhlo.return %677 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x64xf32>
    %277 = "mhlo.fusion"(%276, %269, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x192x4x64xf32>
      mhlo.return %682 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x192x4x64xf32>
    %278 = "mhlo.fusion"(%277, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
      mhlo.return %675 : tensor<1x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
    %279 = "mhlo.fusion"(%278, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
      mhlo.return %675 : tensor<1x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
    %280 = "mhlo.fusion"(%279, %269, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x192x4x64xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x192x4xf32>) -> tensor<192x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x192x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x192xf32>
      mhlo.return %686 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x4x64x192xf32>
    %281 = mhlo.bitcast %280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %282 = "mhlo.fusion"(%arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x192xf32>
      mhlo.return %676 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %283 = mhlo.bitcast %282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %284 = stablehlo.dot %281, %283, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x192xf32>) -> tensor<256x192xf32>
    %285 = mhlo.bitcast %284 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %286 = "mhlo.fusion"(%285, %arg135, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<256x1x192x1x1xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1x192x1x1xf32>) -> tensor<2x8x2x8x1x192xf32>
      %680 = stablehlo.transpose %679, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x1x192xf32>) -> tensor<1x192x2x2x8x8xf32>
      %681 = mhlo.copy %680 : tensor<1x192x2x2x8x8xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x192x2x2x8x8xf32>) -> tensor<1x192x4x64xf32>
      %683 = stablehlo.add %678, %682 : tensor<1x192x4x64xf32>
      mhlo.return %683 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x192xf32>, tensor<192xf32>, tensor<256x1x192x1x1xf32>) -> tensor<1x192x4x64xf32>
    %287 = mhlo.bitcast %286 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %288 = "mhlo.fusion"(%287, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %289 = "mhlo.fusion"(%288, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %290 = "mhlo.fusion"(%289, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %291 = "mhlo.fusion"(%290, %286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %292 = "mhlo.fusion"(%291, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %293 = "mhlo.fusion"(%292, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %294 = "mhlo.fusion"(%293, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %296 = "mhlo.fusion"(%arg136, %295, %290, %286, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %298 = "mhlo.fusion"(%arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x384xf32>
      mhlo.return %676 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %300 = stablehlo.dot %297, %299, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x384xf32>) -> tensor<256x384xf32>
    %301 = mhlo.bitcast %300 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x4x64x384xf32>
    %302 = "mhlo.fusion"(%301, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x384xf32>, %arg246: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x4x64x384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x384xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,384,4,64]{1,3,2,0}"} : (tensor<1x4x64x384xf32>) -> tensor<1x384x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %680 = stablehlo.negate %678 : tensor<1x384x4x64xf32>
      %681 = stablehlo.exponential %680 : tensor<1x384x4x64xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x384x4x64xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x384x4x64xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x384x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,384]{2,1,3,0}"} : (tensor<1x384x4x64xf32>) -> tensor<1x4x64x384xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x384xf32>
      mhlo.return %686 : tensor<1x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x384xf32>, tensor<384xf32>) -> tensor<1x4x64x384xf32>
    %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x384xf32>) -> tensor<256x384xf32>
    %304 = "mhlo.fusion"(%arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x192xf32>
      mhlo.return %676 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %306 = stablehlo.dot %303, %305, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x192xf32>) -> tensor<256x192xf32>
    %307 = mhlo.bitcast %306 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %308 = "mhlo.fusion"(%286, %307, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %311 = "mhlo.fusion"(%310, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %313 = "mhlo.fusion"(%312, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %314 = "mhlo.fusion"(%313, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %315 = "mhlo.fusion"(%314, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %316 = "mhlo.fusion"(%315, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %317 = "mhlo.fusion"(%316) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %318 = "mhlo.fusion"(%arg142, %317, %312, %308, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %319 = mhlo.bitcast %318 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %320 = "mhlo.fusion"(%arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x385xf32>
      mhlo.return %676 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %321 = mhlo.bitcast %320 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %322 = stablehlo.dot %319, %321, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x385xf32>) -> tensor<256x385xf32>
    %323 = mhlo.bitcast %322 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x385xf32>) -> tensor<1x4x64x385xf32>
    %324 = "mhlo.fusion"(%323, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x385xf32>, %arg246: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      mhlo.return %679 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %325 = "mhlo.fusion"(%324, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %326 = "mhlo.fusion"(%325, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %327 = "mhlo.fusion"(%326, %323, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x64xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x64xf32>
      mhlo.return %685 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %328 = "mhlo.fusion"(%327, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %329 = "mhlo.fusion"(%328, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %330 = "mhlo.fusion"(%327, %329) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x64xf32>
      mhlo.return %677 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x64xf32>
    %331 = "mhlo.fusion"(%330, %323, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x192x4x64xf32>
      mhlo.return %682 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x192x4x64xf32>
    %332 = "mhlo.fusion"(%331, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
      mhlo.return %675 : tensor<1x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
    %333 = "mhlo.fusion"(%332, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
      mhlo.return %675 : tensor<1x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
    %334 = "mhlo.fusion"(%333, %323, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x192x4x64xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x192x4xf32>) -> tensor<192x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x192x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x192xf32>
      mhlo.return %686 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x4x64x192xf32>
    %335 = mhlo.bitcast %334 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %336 = "mhlo.fusion"(%arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x192xf32>
      mhlo.return %676 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %337 = mhlo.bitcast %336 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %338 = stablehlo.dot %335, %337, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x192xf32>) -> tensor<256x192xf32>
    %339 = mhlo.bitcast %338 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %340 = "mhlo.fusion"(%308, %339, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %342 = "mhlo.fusion"(%341, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %343 = "mhlo.fusion"(%342, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %344 = "mhlo.fusion"(%343, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %345 = "mhlo.fusion"(%344, %340) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %346 = "mhlo.fusion"(%345, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %347 = "mhlo.fusion"(%346, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %348 = "mhlo.fusion"(%347, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %349 = "mhlo.fusion"(%348) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %350 = "mhlo.fusion"(%arg148, %349, %344, %340, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %351 = mhlo.bitcast %350 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %352 = "mhlo.fusion"(%arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x384xf32>
      mhlo.return %676 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %353 = mhlo.bitcast %352 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %354 = stablehlo.dot %351, %353, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x384xf32>) -> tensor<256x384xf32>
    %355 = mhlo.bitcast %354 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x4x64x384xf32>
    %356 = "mhlo.fusion"(%355, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x384xf32>, %arg246: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x4x64x384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x384xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,384,4,64]{1,3,2,0}"} : (tensor<1x4x64x384xf32>) -> tensor<1x384x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %680 = stablehlo.negate %678 : tensor<1x384x4x64xf32>
      %681 = stablehlo.exponential %680 : tensor<1x384x4x64xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x384x4x64xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x384x4x64xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x384x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,384]{2,1,3,0}"} : (tensor<1x384x4x64xf32>) -> tensor<1x4x64x384xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x384xf32>
      mhlo.return %686 : tensor<1x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x384xf32>, tensor<384xf32>) -> tensor<1x4x64x384xf32>
    %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x384xf32>) -> tensor<256x384xf32>
    %358 = "mhlo.fusion"(%arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x192xf32>
      mhlo.return %676 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %360 = stablehlo.dot %357, %359, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x192xf32>) -> tensor<256x192xf32>
    %361 = mhlo.bitcast %360 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %362 = "mhlo.fusion"(%340, %361, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %363 = mhlo.bitcast %362 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %364 = "mhlo.fusion"(%363, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %365 = "mhlo.fusion"(%364, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %366 = "mhlo.fusion"(%365, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %367 = "mhlo.fusion"(%366, %362) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %368 = "mhlo.fusion"(%367, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %369 = "mhlo.fusion"(%368, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %370 = "mhlo.fusion"(%369, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %371 = "mhlo.fusion"(%370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %372 = "mhlo.fusion"(%arg154, %371, %366, %362, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %373 = mhlo.bitcast %372 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %374 = "mhlo.fusion"(%arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x385xf32>
      mhlo.return %676 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %375 = mhlo.bitcast %374 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %376 = stablehlo.dot %373, %375, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x385xf32>) -> tensor<256x385xf32>
    %377 = mhlo.bitcast %376 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x385xf32>) -> tensor<1x4x64x385xf32>
    %378 = "mhlo.fusion"(%377, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x385xf32>, %arg246: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      mhlo.return %679 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %379 = "mhlo.fusion"(%378, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %380 = "mhlo.fusion"(%379, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %381 = "mhlo.fusion"(%380, %377, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x64xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x64xf32>
      mhlo.return %685 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %382 = "mhlo.fusion"(%381, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %383 = "mhlo.fusion"(%382, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %384 = "mhlo.fusion"(%381, %383) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x64xf32>
      mhlo.return %677 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x64xf32>
    %385 = "mhlo.fusion"(%384, %377, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x192x4x64xf32>
      mhlo.return %682 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x192x4x64xf32>
    %386 = "mhlo.fusion"(%385, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
      mhlo.return %675 : tensor<1x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
    %387 = "mhlo.fusion"(%386, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
      mhlo.return %675 : tensor<1x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
    %388 = "mhlo.fusion"(%387, %377, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x192x4x64xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x192x4xf32>) -> tensor<192x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x192x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x192xf32>
      mhlo.return %686 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x4x64x192xf32>
    %389 = mhlo.bitcast %388 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %390 = "mhlo.fusion"(%arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x192xf32>
      mhlo.return %676 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %391 = mhlo.bitcast %390 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %392 = stablehlo.dot %389, %391, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x192xf32>) -> tensor<256x192xf32>
    %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %394 = "mhlo.fusion"(%362, %393, %arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %395 = mhlo.bitcast %394 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %396 = "mhlo.fusion"(%395, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %397 = "mhlo.fusion"(%396, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %398 = "mhlo.fusion"(%397, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %399 = "mhlo.fusion"(%398, %394) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %400 = "mhlo.fusion"(%399, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %401 = "mhlo.fusion"(%400, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %402 = "mhlo.fusion"(%401, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %403 = "mhlo.fusion"(%402) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %404 = "mhlo.fusion"(%arg160, %403, %398, %394, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %405 = mhlo.bitcast %404 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %406 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x384xf32>
      mhlo.return %676 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %407 = mhlo.bitcast %406 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %408 = stablehlo.dot %405, %407, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x384xf32>) -> tensor<256x384xf32>
    %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x4x64x384xf32>
    %410 = "mhlo.fusion"(%409, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x384xf32>, %arg246: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x4x64x384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x384xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,384,4,64]{1,3,2,0}"} : (tensor<1x4x64x384xf32>) -> tensor<1x384x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %680 = stablehlo.negate %678 : tensor<1x384x4x64xf32>
      %681 = stablehlo.exponential %680 : tensor<1x384x4x64xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x384x4x64xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x384x4x64xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x384x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,384]{2,1,3,0}"} : (tensor<1x384x4x64xf32>) -> tensor<1x4x64x384xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x384xf32>
      mhlo.return %686 : tensor<1x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x384xf32>, tensor<384xf32>) -> tensor<1x4x64x384xf32>
    %411 = mhlo.bitcast %410 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x384xf32>) -> tensor<256x384xf32>
    %412 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x192xf32>
      mhlo.return %676 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %413 = mhlo.bitcast %412 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %414 = stablehlo.dot %411, %413, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x192xf32>) -> tensor<256x192xf32>
    %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %416 = "mhlo.fusion"(%394, %415, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %417 = mhlo.bitcast %416 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %418 = "mhlo.fusion"(%417, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %419 = "mhlo.fusion"(%418, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %420 = "mhlo.fusion"(%419, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %421 = "mhlo.fusion"(%420, %416) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %422 = "mhlo.fusion"(%421, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %423 = "mhlo.fusion"(%422, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %424 = "mhlo.fusion"(%423, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %425 = "mhlo.fusion"(%424) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %426 = "mhlo.fusion"(%arg166, %425, %420, %416, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %428 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x385xf32>
      mhlo.return %676 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %429 = mhlo.bitcast %428 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %430 = stablehlo.dot %427, %429, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x385xf32>) -> tensor<256x385xf32>
    %431 = mhlo.bitcast %430 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x385xf32>) -> tensor<1x4x64x385xf32>
    %432 = "mhlo.fusion"(%431, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x385xf32>, %arg246: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      mhlo.return %679 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %433 = "mhlo.fusion"(%432, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %434 = "mhlo.fusion"(%433, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %435 = "mhlo.fusion"(%434, %431, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x64xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x64xf32>
      mhlo.return %685 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x1x4x64xf32>
    %436 = "mhlo.fusion"(%435, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
      mhlo.return %675 : tensor<1x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<f32>) -> tensor<1x1x4x2xf32>
    %437 = "mhlo.fusion"(%436, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x2xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %438 = "mhlo.fusion"(%435, %437) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x64xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x64xf32>
      mhlo.return %677 : tensor<1x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x64xf32>
    %439 = "mhlo.fusion"(%438, %431, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x64xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 1:193, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x64xf32>) -> tensor<4x64xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x64xf32>) -> tensor<1x192x4x64xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x192x4x64xf32>
      mhlo.return %682 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x64xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x192x4x64xf32>
    %440 = "mhlo.fusion"(%439, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
      mhlo.return %675 : tensor<1x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<f32>) -> tensor<1x192x4x2xf32>
    %441 = "mhlo.fusion"(%440, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
      mhlo.return %675 : tensor<1x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x2xf32>, tensor<f32>) -> tensor<1x192x4xf32>
    %442 = "mhlo.fusion"(%441, %431, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4xf32>, %arg246: tensor<1x4x64x385xf32>, %arg247: tensor<385xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<1x4x64x385xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x385xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,385,4,64]{1,3,2,0}"} : (tensor<1x4x64x385xf32>) -> tensor<1x385x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x385x4x64xf32>
      %679 = stablehlo.slice %678 [0:1, 193:385, 0:4, 0:64] : (tensor<1x385x4x64xf32>) -> tensor<1x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x192x4x64xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x192x4x64xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x192x4xf32>) -> tensor<192x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<192x4xf32>) -> tensor<1x192x4x64xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x192x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x192xf32>
      mhlo.return %686 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4xf32>, tensor<1x4x64x385xf32>, tensor<385xf32>) -> tensor<1x4x64x192xf32>
    %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %444 = "mhlo.fusion"(%arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x192xf32>
      mhlo.return %676 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %446 = stablehlo.dot %443, %445, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x192xf32>) -> tensor<256x192xf32>
    %447 = mhlo.bitcast %446 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %448 = "mhlo.fusion"(%416, %447, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      mhlo.return %679 : tensor<1x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x192x4x64xf32>
    %449 = mhlo.bitcast %448 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
    %450 = "mhlo.fusion"(%449, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %451 = "mhlo.fusion"(%450, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %452 = "mhlo.fusion"(%451, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %453 = "mhlo.fusion"(%452, %448) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x49152xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %681 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>) -> tensor<1x49152xf32>
    %454 = "mhlo.fusion"(%453, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %455 = "mhlo.fusion"(%454, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %456 = "mhlo.fusion"(%455, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %457 = "mhlo.fusion"(%456) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %458 = "mhlo.fusion"(%arg172, %457, %452, %448, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x49152xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x49152xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x192x4x64xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x192x4x64xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,192]{2,1,3,0}"} : (tensor<1x192x4x64xf32>) -> tensor<1x4x64x192xf32>
      %689 = mhlo.copy %688 : tensor<1x4x64x192xf32>
      mhlo.return %689 : tensor<1x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<192xf32>) -> tensor<1x4x64x192xf32>
    %459 = mhlo.bitcast %458 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x192xf32>) -> tensor<256x192xf32>
    %460 = "mhlo.fusion"(%arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x384xf32>
      mhlo.return %676 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %461 = mhlo.bitcast %460 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %462 = stablehlo.dot %459, %461, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x384xf32>) -> tensor<256x384xf32>
    %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x4x64x384xf32>
    %464 = "mhlo.fusion"(%463, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x64x384xf32>, %arg246: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x4x64x384xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x64x384xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,384,4,64]{1,3,2,0}"} : (tensor<1x4x64x384xf32>) -> tensor<1x384x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x384x4x64xf32>
      %680 = stablehlo.negate %678 : tensor<1x384x4x64xf32>
      %681 = stablehlo.exponential %680 : tensor<1x384x4x64xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x384x4x64xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x384x4x64xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x384x4x64xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,64,384]{2,1,3,0}"} : (tensor<1x384x4x64xf32>) -> tensor<1x4x64x384xf32>
      %686 = mhlo.copy %685 : tensor<1x4x64x384xf32>
      mhlo.return %686 : tensor<1x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x64x384xf32>, tensor<384xf32>) -> tensor<1x4x64x384xf32>
    %465 = mhlo.bitcast %464 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x64x384xf32>) -> tensor<256x384xf32>
    %466 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x192xf32>
      mhlo.return %676 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %467 = mhlo.bitcast %466 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %468 = stablehlo.dot %465, %467, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x192xf32>) -> tensor<256x192xf32>
    %469 = mhlo.bitcast %468 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x192xf32>) -> tensor<1x4x64x192xf32>
    %470 = "mhlo.fusion"(%448, %469, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x4x64xf32>, %arg246: tensor<1x4x64x192xf32>, %arg247: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      mhlo.return %680 : tensor<1x1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x1x49152xf32>
    %471 = "mhlo.fusion"(%470, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
      mhlo.return %675 : tensor<1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x49152xf32>, tensor<f32>) -> tensor<1x1x48xf32>
    %472 = "mhlo.fusion"(%471, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
      mhlo.return %675 : tensor<1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x48xf32>, tensor<f32>) -> tensor<1x1x2xf32>
    %473 = "mhlo.fusion"(%472, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x2xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x2xf32>, tensor<f32>) -> tensor<1xf32>
    %474 = "mhlo.fusion"(%473, %448, %469, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x192x4x64xf32>, %arg247: tensor<1x4x64x192xf32>, %arg248: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %676 = stablehlo.add %arg247, %675 : tensor<1x4x64x192xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %678 = mhlo.copy %677 : tensor<1x192x4x64xf32>
      %679 = stablehlo.add %678, %arg246 : tensor<1x192x4x64xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %681 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %684 = stablehlo.subtract %680, %683 : tensor<1x1x49152xf32>
      %685 = stablehlo.multiply %684, %684 : tensor<1x1x49152xf32>
      %686 = mhlo.bitcast %685 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x49152xf32>
      mhlo.return %686 : tensor<1x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>) -> tensor<1x49152xf32>
    %475 = "mhlo.fusion"(%474, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x49152xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
      mhlo.return %675 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x49152xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %476 = "mhlo.fusion"(%475, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1536xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
      mhlo.return %675 : tensor<1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>, tensor<f32>) -> tensor<1x48xf32>
    %477 = "mhlo.fusion"(%476, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x48xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 8]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
      mhlo.return %675 : tensor<1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x48xf32>, tensor<f32>) -> tensor<1x2xf32>
    %478 = "mhlo.fusion"(%477) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x2xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2xf32>) -> tensor<1xf32>
    %479 = "mhlo.fusion"(%arg178, %478, %473, %448, %469, %arg177, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x192x4x64xf32>, %arg249: tensor<1x4x64x192xf32>, %arg250: tensor<192xf32>, %arg251: tensor<192xf32>):
      %675 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %676 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<192xf32>) -> tensor<1x4x64x192xf32>
      %677 = stablehlo.add %arg249, %676 : tensor<1x4x64x192xf32>
      %678 = stablehlo.transpose %677, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,192,4,64]{1,3,2,0}"} : (tensor<1x4x64x192xf32>) -> tensor<1x192x4x64xf32>
      %679 = mhlo.copy %678 : tensor<1x192x4x64xf32>
      %680 = stablehlo.add %679, %arg248 : tensor<1x192x4x64xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x1x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<1xf32>
      %682 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %683 = mhlo.bitcast %682 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %684 = stablehlo.broadcast_in_dim %683, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %685 = stablehlo.subtract %681, %684 : tensor<1x1x49152xf32>
      %686 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %687 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<f32>) -> tensor<1x1x49152xf32>
      %688 = stablehlo.multiply %685, %687 : tensor<1x1x49152xf32>
      %689 = mhlo.bitcast %688 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x49152xf32>) -> tensor<1x192x4x64xf32>
      %690 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<1x192x4x64xf32>
      %691 = stablehlo.multiply %689, %690 : tensor<1x192x4x64xf32>
      %692 = stablehlo.add %675, %691 : tensor<1x192x4x64xf32>
      %693 = mhlo.bitcast %692 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x4x64xf32>) -> tensor<1x192x2x2x8x8xf32>
      %694 = stablehlo.transpose %693, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,8,2,8,1,192]{3,1,2,0,5,4}"} : (tensor<1x192x2x2x8x8xf32>) -> tensor<2x8x2x8x1x192xf32>
      %695 = mhlo.copy %694 : tensor<2x8x2x8x1x192xf32>
      %696 = mhlo.bitcast %695 {result_layout = dense<[2, 0, 4, 3, 1]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[256,1,192,1,1]{2,0,4,3,1}"} : (tensor<2x8x2x8x1x192xf32>) -> tensor<256x1x192x1x1xf32>
      %697 = mhlo.copy %696 : tensor<256x1x192x1x1xf32>
      mhlo.return %697 : tensor<256x1x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x192x4x64xf32>, tensor<1x4x64x192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<256x1x192x1x1xf32>
    %480 = "mhlo.fusion"(%8, %9, %479) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x16x16xf32>, %arg246: tensor<256x2xi32>, %arg247: tensor<256x1x192x1x1xf32>):
      %675 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %676 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1x192x1x1xf32>) -> tensor<1x192x16x16xf32>
      mhlo.return %675 : tensor<1x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1x192x1x1xf32>) -> tensor<1x192x16x16xf32>
    %481 = "mhlo.fusion"(%480) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x192x16x16xf32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,192]{2,1,3,0}"} : (tensor<1x192x16x16xf32>) -> tensor<1x16x16x192xf32>
      %676 = mhlo.copy %675 : tensor<1x16x16x192xf32>
      mhlo.return %676 : tensor<1x16x16x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x16x16xf32>) -> tensor<1x16x16x192xf32>
    %482 = mhlo.bitcast %481 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x192xf32>) -> tensor<256x192xf32>
    %483 = "mhlo.fusion"(%arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %676 = mhlo.copy %675 : tensor<1x1x192x384xf32>
      mhlo.return %676 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %484 = mhlo.bitcast %483 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %485 = stablehlo.dot %482, %484, precision = [DEFAULT, DEFAULT] : (tensor<256x192xf32>, tensor<192x384xf32>) -> tensor<256x384xf32>
    %486 = mhlo.bitcast %485 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x384xf32>) -> tensor<1x16x16x384xf32>
    %487 = "mhlo.fusion"(%arg181, %arg182, %7, %486, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1x16x16x384xf32>, %arg249: tensor<384xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x16x16x384xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x16x16x384xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x16x16x384xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<1x16x16x384xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x16x16x384xf32>
      mhlo.return %682 : tensor<1x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1x16x16x384xf32>, tensor<384xf32>) -> tensor<1x16x16x384xf32>
    %488 = mhlo.bitcast %487 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x384xf32>) -> tensor<256x384xf32>
    %489 = "mhlo.fusion"(%arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x384x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,768]{1,0,2,3}"} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
      %676 = mhlo.copy %675 : tensor<1x1x384x768xf32>
      mhlo.return %676 : tensor<1x1x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
    %490 = mhlo.bitcast %489 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x768xf32>) -> tensor<384x768xf32>
    %491 = stablehlo.dot %488, %490, precision = [DEFAULT, DEFAULT] : (tensor<256x384xf32>, tensor<384x768xf32>) -> tensor<256x768xf32>
    %492 = mhlo.bitcast %491 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x768xf32>) -> tensor<1x16x16x768xf32>
    %493 = "mhlo.fusion"(%arg185, %arg186, %6, %492, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<1x16x16x768xf32>, %arg249: tensor<768xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<1x16x16x768xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x16x16x768xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<1x16x16x768xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x16x16x768xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<1x16x16x768xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x16x16x768xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<1x16x16x768xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x16x16x768xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,16,16]{1,3,2,0}"} : (tensor<1x16x16x768xf32>) -> tensor<1x768x16x16xf32>
      %684 = mhlo.copy %683 : tensor<1x768x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x768x16x16xf32>
      %686 = stablehlo.negate %684 : tensor<1x768x16x16xf32>
      %687 = stablehlo.exponential %686 : tensor<1x768x16x16xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x768x16x16xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x768x16x16xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x768x16x16xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,768]{2,1,3,0}"} : (tensor<1x768x16x16xf32>) -> tensor<1x16x16x768xf32>
      %692 = mhlo.copy %691 : tensor<1x16x16x768xf32>
      mhlo.return %692 : tensor<1x16x16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1x16x16x768xf32>, tensor<768xf32>) -> tensor<1x16x16x768xf32>
    %494 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,768]{1,0,2,3}"} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x768xf32>
      mhlo.return %676 : tensor<3x3x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
    %495 = stablehlo.convolution(%493, %494) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x16x16x768xf32>, tensor<3x3x1x768xf32>) -> tensor<1x8x8x768xf32>
    %496 = "mhlo.fusion"(%arg189, %arg190, %5, %495, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<1x8x8x768xf32>, %arg249: tensor<768xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<1x8x8x768xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x8x8x768xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<1x8x8x768xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x8x8x768xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<1x8x8x768xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x8x8x768xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<1x8x8x768xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x8x8x768xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,8,8]{1,3,2,0}"} : (tensor<1x8x8x768xf32>) -> tensor<1x768x8x8xf32>
      %684 = mhlo.copy %683 : tensor<1x768x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x768x8x8xf32>
      %686 = stablehlo.negate %684 : tensor<1x768x8x8xf32>
      %687 = stablehlo.exponential %686 : tensor<1x768x8x8xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x768x8x8xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x768x8x8xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x768x8x8xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,8,768]{2,1,3,0}"} : (tensor<1x768x8x8xf32>) -> tensor<1x8x8x768xf32>
      %692 = mhlo.copy %691 : tensor<1x8x8x768xf32>
      mhlo.return %692 : tensor<1x8x8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1x8x8x768xf32>, tensor<768xf32>) -> tensor<1x8x8x768xf32>
    %497 = mhlo.bitcast %496 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x8x768xf32>) -> tensor<64x768xf32>
    %498 = "mhlo.fusion"(%arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x768x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,768,512]{1,0,2,3}"} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x768x512xf32>
      mhlo.return %676 : tensor<1x1x768x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
    %499 = mhlo.bitcast %498 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x768x512xf32>) -> tensor<768x512xf32>
    %500 = stablehlo.dot %497, %499, precision = [DEFAULT, DEFAULT] : (tensor<64x768xf32>, tensor<768x512xf32>) -> tensor<64x512xf32>
    %501 = mhlo.bitcast %500 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x512xf32>) -> tensor<1x8x8x512xf32>
    %502 = "mhlo.fusion"(%arg193, %arg194, %4, %501, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1x8x8x512xf32>, %arg249: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x8x8x512xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x8x8x512xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x8x8x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x8x8x512xf32>
      mhlo.return %682 : tensor<1x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x8x8x512xf32>, tensor<512xf32>) -> tensor<1x8x8x512xf32>
    %503 = "mhlo.fusion"(%arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %676 = mhlo.copy %675 : tensor<3x3x1x512xf32>
      mhlo.return %676 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %504 = stablehlo.convolution(%502, %503) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x8x8x512xf32>, tensor<3x3x1x512xf32>) -> tensor<1x8x8x512xf32>
    %505 = "mhlo.fusion"(%arg197, %arg198, %3, %504, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1x8x8x512xf32>, %arg249: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x8x8x512xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x8x8x512xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x8x8x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x8x8x512xf32>
      %683 = stablehlo.transpose %682, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,8,8]{1,3,2,0}"} : (tensor<1x8x8x512xf32>) -> tensor<1x512x8x8xf32>
      %684 = mhlo.copy %683 : tensor<1x512x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x8x8xf32>
      %686 = stablehlo.negate %684 : tensor<1x512x8x8xf32>
      %687 = stablehlo.exponential %686 : tensor<1x512x8x8xf32>
      %688 = stablehlo.add %685, %687 : tensor<1x512x8x8xf32>
      %689 = stablehlo.divide %685, %688 : tensor<1x512x8x8xf32>
      %690 = stablehlo.multiply %684, %689 : tensor<1x512x8x8xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,8,512]{2,1,3,0}"} : (tensor<1x512x8x8xf32>) -> tensor<1x8x8x512xf32>
      %692 = mhlo.copy %691 : tensor<1x8x8x512xf32>
      mhlo.return %692 : tensor<1x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x8x8x512xf32>, tensor<512xf32>) -> tensor<1x8x8x512xf32>
    %506 = mhlo.bitcast %505 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x8x512xf32>) -> tensor<64x512xf32>
    %507 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x512x256xf32>
      mhlo.return %676 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %508 = mhlo.bitcast %507 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %509 = stablehlo.dot %506, %508, precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x256xf32>) -> tensor<64x256xf32>
    %510 = mhlo.bitcast %509 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x8x8x256xf32>
    %511 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %675 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %677 = stablehlo.multiply %675, %676 : tensor<4xi32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %679 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %680 = stablehlo.add %678, %679 : tensor<2x4xi32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %682 = mhlo.copy %681 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %683 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %684 = stablehlo.compare LT, %682, %683 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %685 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %686 = stablehlo.add %682, %685 : tensor<2x4x1x1xi32>
      %687 = stablehlo.select %684, %686, %682 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %688 = mhlo.bitcast %687 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %690 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %691 = stablehlo.compare LT, %680, %690 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %692 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %693 = stablehlo.add %680, %692 : tensor<2x4xi32>
      %694 = stablehlo.select %691, %693, %680 : tensor<2x4xi1>, tensor<2x4xi32>
      %695 = stablehlo.broadcast_in_dim %694, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %696 = stablehlo.concatenate %689, %695, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      mhlo.return %696 : tensor<2x4x2x4x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x4x2x4x2xi32>
    %512 = "mhlo.fusion"(%510, %511) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x8x8x256xf32>, %arg246: tensor<2x4x2x4x2xi32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,8,8]{1,3,2,0}"} : (tensor<1x8x8x256xf32>) -> tensor<1x256x8x8xf32>
      %676 = mhlo.copy %675 : tensor<1x256x8x8xf32>
      %677 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %678 = "stablehlo.gather"(%676, %677) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 1, 1>}> : (tensor<1x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1x256x1x1xf32>
      mhlo.return %678 : tensor<64x1x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x8x256xf32>, tensor<2x4x2x4x2xi32>) -> tensor<64x1x256x1x1xf32>
    %513 = "mhlo.fusion"(%512) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x1x256x1x1xf32>):
      %675 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<2x4x2x4x1x256xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x1x256xf32>) -> tensor<1x256x2x2x4x4xf32>
      %677 = mhlo.copy %676 : tensor<1x256x2x2x4x4xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x1x16384xf32>
      mhlo.return %678 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x256x1x1xf32>) -> tensor<1x1x16384xf32>
    %514 = "mhlo.fusion"(%513, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %515 = "mhlo.fusion"(%514, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %516 = "mhlo.fusion"(%515, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %517 = "mhlo.fusion"(%516, %512) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<64x1x256x1x1xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<2x4x2x4x1x256xf32>
      %676 = stablehlo.transpose %675, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x1x256xf32>) -> tensor<1x256x2x2x4x4xf32>
      %677 = mhlo.copy %676 : tensor<1x256x2x2x4x4xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %679 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %682 = stablehlo.subtract %678, %681 : tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %682, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %684 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<64x1x256x1x1xf32>) -> tensor<1x16384xf32>
    %518 = "mhlo.fusion"(%517, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %519 = "mhlo.fusion"(%518, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %520 = "mhlo.fusion"(%519) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %521 = "mhlo.fusion"(%arg201, %520, %516, %512, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<64x1x256x1x1xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<2x4x2x4x1x256xf32>
      %677 = stablehlo.transpose %676, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x1x256xf32>) -> tensor<1x256x2x2x4x4xf32>
      %678 = mhlo.copy %677 : tensor<1x256x2x2x4x4xf32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %680 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.subtract %679, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %685 = stablehlo.broadcast_in_dim %684, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %686 = stablehlo.multiply %683, %685 : tensor<1x1x16384xf32>
      %687 = mhlo.bitcast %686 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %688 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %689 = stablehlo.multiply %687, %688 : tensor<1x256x4x16xf32>
      %690 = stablehlo.add %675, %689 : tensor<1x256x4x16xf32>
      %691 = stablehlo.transpose %690, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %692 = mhlo.copy %691 : tensor<1x4x16x256xf32>
      mhlo.return %692 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x1x256x1x1xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %522 = mhlo.bitcast %521 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %523 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x513xf32>
      mhlo.return %676 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %525 = stablehlo.dot %522, %524, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x513xf32>) -> tensor<64x513xf32>
    %526 = mhlo.bitcast %525 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x513xf32>) -> tensor<1x4x16x513xf32>
    %527 = "mhlo.fusion"(%526, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x513xf32>, %arg246: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.reduce(%679 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %680 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4xf32>
    %528 = "mhlo.fusion"(%527, %526, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x16xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x16xf32>
      mhlo.return %685 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4x16xf32>
    %529 = "mhlo.fusion"(%528, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %530 = "mhlo.fusion"(%528, %529) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x16xf32>
      mhlo.return %677 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x16xf32>
    %531 = "mhlo.fusion"(%530, %526, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %683 = stablehlo.reduce(%682 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4xf32>
      mhlo.return %683 : tensor<1x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x256x4xf32>
    %532 = "mhlo.fusion"(%531, %526, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x256x4x16xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4xf32>) -> tensor<256x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x256x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x256xf32>
      mhlo.return %686 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x4x16x256xf32>
    %533 = mhlo.bitcast %532 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %534 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x256xf32>
      mhlo.return %676 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %535 = mhlo.bitcast %534 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %536 = stablehlo.dot %533, %535, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x256xf32>) -> tensor<64x256xf32>
    %537 = mhlo.bitcast %536 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %538 = "mhlo.fusion"(%537, %arg206, %512) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x1x256x1x1xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1x256x1x1xf32>) -> tensor<2x4x2x4x1x256xf32>
      %680 = stablehlo.transpose %679, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[1,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x1x256xf32>) -> tensor<1x256x2x2x4x4xf32>
      %681 = mhlo.copy %680 : tensor<1x256x2x2x4x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1x256x2x2x4x4xf32>) -> tensor<1x256x4x16xf32>
      %683 = stablehlo.add %678, %682 : tensor<1x256x4x16xf32>
      mhlo.return %683 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x256xf32>, tensor<256xf32>, tensor<64x1x256x1x1xf32>) -> tensor<1x256x4x16xf32>
    %539 = mhlo.bitcast %538 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %540 = "mhlo.fusion"(%539, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %541 = "mhlo.fusion"(%540, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %542 = "mhlo.fusion"(%541, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %543 = "mhlo.fusion"(%542, %538) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x16384xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %681 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>) -> tensor<1x16384xf32>
    %544 = "mhlo.fusion"(%543, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %545 = "mhlo.fusion"(%544, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %546 = "mhlo.fusion"(%545) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %547 = "mhlo.fusion"(%arg207, %546, %542, %538, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x256x4x16xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x256x4x16xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %689 = mhlo.copy %688 : tensor<1x4x16x256xf32>
      mhlo.return %689 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %548 = mhlo.bitcast %547 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %549 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x512xf32>
      mhlo.return %676 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %550 = mhlo.bitcast %549 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %551 = stablehlo.dot %548, %550, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
    %552 = mhlo.bitcast %551 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x512xf32>) -> tensor<1x4x16x512xf32>
    %553 = "mhlo.fusion"(%552, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x512xf32>, %arg246: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x4x16x512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x512xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,4,16]{1,3,2,0}"} : (tensor<1x4x16x512xf32>) -> tensor<1x512x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %680 = stablehlo.negate %678 : tensor<1x512x4x16xf32>
      %681 = stablehlo.exponential %680 : tensor<1x512x4x16xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x512x4x16xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x512x4x16xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x512x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,512]{2,1,3,0}"} : (tensor<1x512x4x16xf32>) -> tensor<1x4x16x512xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x512xf32>
      mhlo.return %686 : tensor<1x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x512xf32>, tensor<512xf32>) -> tensor<1x4x16x512xf32>
    %554 = mhlo.bitcast %553 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x512xf32>) -> tensor<64x512xf32>
    %555 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x512x256xf32>
      mhlo.return %676 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %556 = mhlo.bitcast %555 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %557 = stablehlo.dot %554, %556, precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x256xf32>) -> tensor<64x256xf32>
    %558 = mhlo.bitcast %557 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %559 = "mhlo.fusion"(%538, %558, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<1x4x16x256xf32>, %arg247: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x256x4x16xf32>
      mhlo.return %679 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %560 = mhlo.bitcast %559 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %561 = "mhlo.fusion"(%560, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %562 = "mhlo.fusion"(%561, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %563 = "mhlo.fusion"(%562, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %564 = "mhlo.fusion"(%563, %559) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x16384xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %681 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>) -> tensor<1x16384xf32>
    %565 = "mhlo.fusion"(%564, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %566 = "mhlo.fusion"(%565, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %567 = "mhlo.fusion"(%566) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %568 = "mhlo.fusion"(%arg213, %567, %563, %559, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x256x4x16xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x256x4x16xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %689 = mhlo.copy %688 : tensor<1x4x16x256xf32>
      mhlo.return %689 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %569 = mhlo.bitcast %568 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %570 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x513xf32>
      mhlo.return %676 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %571 = mhlo.bitcast %570 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %572 = stablehlo.dot %569, %571, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x513xf32>) -> tensor<64x513xf32>
    %573 = mhlo.bitcast %572 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x513xf32>) -> tensor<1x4x16x513xf32>
    %574 = "mhlo.fusion"(%573, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x513xf32>, %arg246: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.reduce(%679 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %680 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4xf32>
    %575 = "mhlo.fusion"(%574, %573, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x16xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x16xf32>
      mhlo.return %685 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4x16xf32>
    %576 = "mhlo.fusion"(%575, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %577 = "mhlo.fusion"(%575, %576) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x16xf32>
      mhlo.return %677 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x16xf32>
    %578 = "mhlo.fusion"(%577, %573, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %683 = stablehlo.reduce(%682 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4xf32>
      mhlo.return %683 : tensor<1x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x256x4xf32>
    %579 = "mhlo.fusion"(%578, %573, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x256x4x16xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4xf32>) -> tensor<256x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x256x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x256xf32>
      mhlo.return %686 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x4x16x256xf32>
    %580 = mhlo.bitcast %579 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %581 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x256xf32>
      mhlo.return %676 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %582 = mhlo.bitcast %581 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %583 = stablehlo.dot %580, %582, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x256xf32>) -> tensor<64x256xf32>
    %584 = mhlo.bitcast %583 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %585 = "mhlo.fusion"(%559, %584, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<1x4x16x256xf32>, %arg247: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x256x4x16xf32>
      mhlo.return %679 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %586 = mhlo.bitcast %585 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %587 = "mhlo.fusion"(%586, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %588 = "mhlo.fusion"(%587, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %589 = "mhlo.fusion"(%588, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %590 = "mhlo.fusion"(%589, %585) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x16384xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %681 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>) -> tensor<1x16384xf32>
    %591 = "mhlo.fusion"(%590, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %592 = "mhlo.fusion"(%591, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %593 = "mhlo.fusion"(%592) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %594 = "mhlo.fusion"(%arg219, %593, %589, %585, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x256x4x16xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x256x4x16xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %689 = mhlo.copy %688 : tensor<1x4x16x256xf32>
      mhlo.return %689 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %595 = mhlo.bitcast %594 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %596 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x512xf32>
      mhlo.return %676 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %597 = mhlo.bitcast %596 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %598 = stablehlo.dot %595, %597, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
    %599 = mhlo.bitcast %598 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x512xf32>) -> tensor<1x4x16x512xf32>
    %600 = "mhlo.fusion"(%599, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x512xf32>, %arg246: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x4x16x512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x512xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,4,16]{1,3,2,0}"} : (tensor<1x4x16x512xf32>) -> tensor<1x512x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %680 = stablehlo.negate %678 : tensor<1x512x4x16xf32>
      %681 = stablehlo.exponential %680 : tensor<1x512x4x16xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x512x4x16xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x512x4x16xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x512x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,512]{2,1,3,0}"} : (tensor<1x512x4x16xf32>) -> tensor<1x4x16x512xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x512xf32>
      mhlo.return %686 : tensor<1x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x512xf32>, tensor<512xf32>) -> tensor<1x4x16x512xf32>
    %601 = mhlo.bitcast %600 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x512xf32>) -> tensor<64x512xf32>
    %602 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x512x256xf32>
      mhlo.return %676 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %603 = mhlo.bitcast %602 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %604 = stablehlo.dot %601, %603, precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x256xf32>) -> tensor<64x256xf32>
    %605 = mhlo.bitcast %604 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %606 = "mhlo.fusion"(%585, %605, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<1x4x16x256xf32>, %arg247: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x256x4x16xf32>
      mhlo.return %679 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %607 = mhlo.bitcast %606 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %608 = "mhlo.fusion"(%607, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %609 = "mhlo.fusion"(%608, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %610 = "mhlo.fusion"(%609, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %611 = "mhlo.fusion"(%610, %606) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x16384xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %681 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>) -> tensor<1x16384xf32>
    %612 = "mhlo.fusion"(%611, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %613 = "mhlo.fusion"(%612, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %614 = "mhlo.fusion"(%613) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %615 = "mhlo.fusion"(%arg225, %614, %610, %606, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x256x4x16xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x256x4x16xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %689 = mhlo.copy %688 : tensor<1x4x16x256xf32>
      mhlo.return %689 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %616 = mhlo.bitcast %615 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %617 = "mhlo.fusion"(%arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x513xf32>
      mhlo.return %676 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %618 = mhlo.bitcast %617 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %619 = stablehlo.dot %616, %618, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x513xf32>) -> tensor<64x513xf32>
    %620 = mhlo.bitcast %619 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x513xf32>) -> tensor<1x4x16x513xf32>
    %621 = "mhlo.fusion"(%620, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x513xf32>, %arg246: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.reduce(%679 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %680 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4xf32>
    %622 = "mhlo.fusion"(%621, %620, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 0:1, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
      %681 = stablehlo.maximum %680, %arg245 : tensor<1x1x4xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %684 = stablehlo.subtract %679, %683 : tensor<1x1x4x16xf32>
      %685 = stablehlo.exponential %684 : tensor<1x1x4x16xf32>
      mhlo.return %685 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x1x4x16xf32>
    %623 = "mhlo.fusion"(%622, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
      mhlo.return %675 : tensor<1x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %624 = "mhlo.fusion"(%622, %623) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x1x4xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x4xf32>) -> tensor<4xf32>
      %676 = stablehlo.broadcast_in_dim %675, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4x16xf32>
      %677 = stablehlo.divide %arg245, %676 : tensor<1x1x4x16xf32>
      mhlo.return %677 : tensor<1x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4x16xf32>
    %625 = "mhlo.fusion"(%624, %620, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x4x16xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 1:257, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %680 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x4x16xf32>) -> tensor<4x16xf32>
      %681 = stablehlo.broadcast_in_dim %680, dims = [2, 3] : (tensor<4x16xf32>) -> tensor<1x256x4x16xf32>
      %682 = stablehlo.multiply %679, %681 : tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %683 = stablehlo.reduce(%682 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<1x256x4x16xf32>, tensor<f32>) -> tensor<1x256x4xf32>
      mhlo.return %683 : tensor<1x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x4x16xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x256x4xf32>
    %626 = "mhlo.fusion"(%625, %620, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4xf32>, %arg246: tensor<1x4x16x513xf32>, %arg247: tensor<513xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<1x4x16x513xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x513xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,513,4,16]{1,3,2,0}"} : (tensor<1x4x16x513xf32>) -> tensor<1x513x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x513x4x16xf32>
      %679 = stablehlo.slice %678 [0:1, 257:513, 0:4, 0:16] : (tensor<1x513x4x16xf32>) -> tensor<1x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %680 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x256x4x16xf32>
      %681 = stablehlo.maximum %679, %680 : tensor<1x256x4x16xf32>
      %682 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x256x4xf32>) -> tensor<256x4xf32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [1, 2] : (tensor<256x4xf32>) -> tensor<1x256x4x16xf32>
      %684 = stablehlo.multiply %681, %683 : tensor<1x256x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x256xf32>
      mhlo.return %686 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4xf32>, tensor<1x4x16x513xf32>, tensor<513xf32>) -> tensor<1x4x16x256xf32>
    %627 = mhlo.bitcast %626 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %628 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x256xf32>
      mhlo.return %676 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %629 = mhlo.bitcast %628 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %630 = stablehlo.dot %627, %629, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x256xf32>) -> tensor<64x256xf32>
    %631 = mhlo.bitcast %630 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %632 = "mhlo.fusion"(%606, %631, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<1x4x16x256xf32>, %arg247: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x256x4x16xf32>
      mhlo.return %679 : tensor<1x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x256x4x16xf32>
    %633 = mhlo.bitcast %632 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
    %634 = "mhlo.fusion"(%633, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %635 = "mhlo.fusion"(%634, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %636 = "mhlo.fusion"(%635, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %637 = "mhlo.fusion"(%636, %632) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>):
      %675 = mhlo.bitcast %arg246 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %679 = stablehlo.subtract %675, %678 : tensor<1x1x16384xf32>
      %680 = stablehlo.multiply %679, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %681 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>) -> tensor<1x16384xf32>
    %638 = "mhlo.fusion"(%637, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %639 = "mhlo.fusion"(%638, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %640 = "mhlo.fusion"(%639) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %641 = "mhlo.fusion"(%arg231, %640, %636, %632, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = mhlo.bitcast %arg248 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %677 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %678 = mhlo.bitcast %677 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %679 = stablehlo.broadcast_in_dim %678, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %680 = stablehlo.subtract %676, %679 : tensor<1x1x16384xf32>
      %681 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %682 = stablehlo.broadcast_in_dim %681, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %683 = stablehlo.multiply %680, %682 : tensor<1x1x16384xf32>
      %684 = mhlo.bitcast %683 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %685 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %686 = stablehlo.multiply %684, %685 : tensor<1x256x4x16xf32>
      %687 = stablehlo.add %675, %686 : tensor<1x256x4x16xf32>
      %688 = stablehlo.transpose %687, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,256]{2,1,3,0}"} : (tensor<1x256x4x16xf32>) -> tensor<1x4x16x256xf32>
      %689 = mhlo.copy %688 : tensor<1x4x16x256xf32>
      mhlo.return %689 : tensor<1x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<256xf32>) -> tensor<1x4x16x256xf32>
    %642 = mhlo.bitcast %641 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x256xf32>) -> tensor<64x256xf32>
    %643 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x512xf32>
      mhlo.return %676 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %644 = mhlo.bitcast %643 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %645 = stablehlo.dot %642, %644, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
    %646 = mhlo.bitcast %645 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x512xf32>) -> tensor<1x4x16x512xf32>
    %647 = "mhlo.fusion"(%646, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x4x16x512xf32>, %arg246: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x4x16x512xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<1x4x16x512xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,512,4,16]{1,3,2,0}"} : (tensor<1x4x16x512xf32>) -> tensor<1x512x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512x4x16xf32>
      %680 = stablehlo.negate %678 : tensor<1x512x4x16xf32>
      %681 = stablehlo.exponential %680 : tensor<1x512x4x16xf32>
      %682 = stablehlo.add %679, %681 : tensor<1x512x4x16xf32>
      %683 = stablehlo.divide %679, %682 : tensor<1x512x4x16xf32>
      %684 = stablehlo.multiply %678, %683 : tensor<1x512x4x16xf32>
      %685 = stablehlo.transpose %684, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,512]{2,1,3,0}"} : (tensor<1x512x4x16xf32>) -> tensor<1x4x16x512xf32>
      %686 = mhlo.copy %685 : tensor<1x4x16x512xf32>
      mhlo.return %686 : tensor<1x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x16x512xf32>, tensor<512xf32>) -> tensor<1x4x16x512xf32>
    %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x512xf32>) -> tensor<64x512xf32>
    %649 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %676 = mhlo.copy %675 : tensor<1x1x512x256xf32>
      mhlo.return %676 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %650 = mhlo.bitcast %649 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %651 = stablehlo.dot %648, %650, precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x256xf32>) -> tensor<64x256xf32>
    %652 = mhlo.bitcast %651 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x256xf32>) -> tensor<1x4x16x256xf32>
    %653 = "mhlo.fusion"(%632, %652, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x4x16xf32>, %arg246: tensor<1x4x16x256xf32>, %arg247: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg246, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg245 : tensor<1x256x4x16xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      mhlo.return %680 : tensor<1x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x1x16384xf32>
    %654 = "mhlo.fusion"(%653, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
      mhlo.return %675 : tensor<1x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16384xf32>, tensor<f32>) -> tensor<1x1x512xf32>
    %655 = "mhlo.fusion"(%654, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
      mhlo.return %675 : tensor<1x1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1x16xf32>
    %656 = "mhlo.fusion"(%655, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x1x16xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %675 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x16xf32>, tensor<f32>) -> tensor<1xf32>
    %657 = "mhlo.fusion"(%656, %632, %652, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1xf32>, %arg246: tensor<1x256x4x16xf32>, %arg247: tensor<1x4x16x256xf32>, %arg248: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %676 = stablehlo.add %arg247, %675 : tensor<1x4x16x256xf32>
      %677 = stablehlo.transpose %676, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %678 = mhlo.copy %677 : tensor<1x256x4x16xf32>
      %679 = stablehlo.add %678, %arg246 : tensor<1x256x4x16xf32>
      %680 = mhlo.bitcast %679 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %681 = stablehlo.multiply %arg245, %cst_1 : tensor<1xf32>
      %682 = mhlo.bitcast %681 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %683 = stablehlo.broadcast_in_dim %682, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %684 = stablehlo.subtract %680, %683 : tensor<1x1x16384xf32>
      %685 = stablehlo.multiply %684, %684 : tensor<1x1x16384xf32>
      %686 = mhlo.bitcast %685 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x16384xf32>
      mhlo.return %686 : tensor<1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>) -> tensor<1x16384xf32>
    %658 = "mhlo.fusion"(%657, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16384xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16384xf32>, tensor<f32>) -> tensor<1x512xf32>
    %659 = "mhlo.fusion"(%658, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>, %arg246: tensor<f32>):
      %675 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %676 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
      mhlo.return %675 : tensor<1x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>, tensor<f32>) -> tensor<1x16xf32>
    %660 = "mhlo.fusion"(%659) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x16xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %675 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x16xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_2 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %676 = stablehlo.multiply %675, %cst_2 : tensor<1xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
      %677 = stablehlo.add %676, %cst_3 : tensor<1xf32>
      %678 = stablehlo.rsqrt %677 : tensor<1xf32>
      mhlo.return %678 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16xf32>) -> tensor<1xf32>
    %661 = "mhlo.fusion"(%arg237, %660, %656, %632, %652, %arg236, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1xf32>, %arg247: tensor<1xf32>, %arg248: tensor<1x256x4x16xf32>, %arg249: tensor<1x4x16x256xf32>, %arg250: tensor<256xf32>, %arg251: tensor<256xf32>):
      %675 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %676 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<256xf32>) -> tensor<1x4x16x256xf32>
      %677 = stablehlo.add %arg249, %676 : tensor<1x4x16x256xf32>
      %678 = stablehlo.transpose %677, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,256,4,16]{1,3,2,0}"} : (tensor<1x4x16x256xf32>) -> tensor<1x256x4x16xf32>
      %679 = mhlo.copy %678 : tensor<1x256x4x16xf32>
      %680 = stablehlo.add %679, %arg248 : tensor<1x256x4x16xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x1x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<1xf32>
      %682 = stablehlo.multiply %arg247, %cst_1 : tensor<1xf32>
      %683 = mhlo.bitcast %682 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %684 = stablehlo.broadcast_in_dim %683, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %685 = stablehlo.subtract %681, %684 : tensor<1x1x16384xf32>
      %686 = mhlo.bitcast %arg246 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %687 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<f32>) -> tensor<1x1x16384xf32>
      %688 = stablehlo.multiply %685, %687 : tensor<1x1x16384xf32>
      %689 = mhlo.bitcast %688 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x16384xf32>) -> tensor<1x256x4x16xf32>
      %690 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1x256x4x16xf32>
      %691 = stablehlo.multiply %689, %690 : tensor<1x256x4x16xf32>
      %692 = stablehlo.add %675, %691 : tensor<1x256x4x16xf32>
      %693 = mhlo.bitcast %692 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x256x4x16xf32>) -> tensor<1x256x2x2x4x4xf32>
      %694 = stablehlo.transpose %693, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,4,2,4,1,256]{3,1,2,0,5,4}"} : (tensor<1x256x2x2x4x4xf32>) -> tensor<2x4x2x4x1x256xf32>
      %695 = mhlo.copy %694 : tensor<2x4x2x4x1x256xf32>
      %696 = mhlo.bitcast %695 {result_layout = dense<[2, 0, 4, 3, 1]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[64,1,256,1,1]{2,0,4,3,1}"} : (tensor<2x4x2x4x1x256xf32>) -> tensor<64x1x256x1x1xf32>
      %697 = mhlo.copy %696 : tensor<64x1x256x1x1xf32>
      mhlo.return %697 : tensor<64x1x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x256x4x16xf32>, tensor<1x4x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x1x256x1x1xf32>
    %662 = "mhlo.fusion"(%1, %2, %661) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x8x8xf32>, %arg246: tensor<64x2xi32>, %arg247: tensor<64x1x256x1x1xf32>):
      %675 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %676 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %676 : tensor<f32>
      }) : (tensor<1x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1x256x1x1xf32>) -> tensor<1x256x8x8xf32>
      mhlo.return %675 : tensor<1x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1x256x1x1xf32>) -> tensor<1x256x8x8xf32>
    %663 = "mhlo.fusion"(%662) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x256x8x8xf32>):
      %675 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,8,256]{2,1,3,0}"} : (tensor<1x256x8x8xf32>) -> tensor<1x8x8x256xf32>
      %676 = mhlo.copy %675 : tensor<1x8x8x256xf32>
      mhlo.return %676 : tensor<1x8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x8x8xf32>) -> tensor<1x8x8x256xf32>
    %664 = mhlo.bitcast %663 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x8x256xf32>) -> tensor<64x256xf32>
    %665 = "mhlo.fusion"(%arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %675 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %676 = mhlo.copy %675 : tensor<1x1x256x512xf32>
      mhlo.return %676 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %666 = mhlo.bitcast %665 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %667 = stablehlo.dot %664, %666, precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x512xf32>) -> tensor<64x512xf32>
    %668 = mhlo.bitcast %667 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<64x512xf32>) -> tensor<1x8x8x512xf32>
    %669 = "mhlo.fusion"(%arg240, %arg241, %0, %668, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1x8x8x512xf32>, %arg249: tensor<512xf32>):
      %675 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %676 = stablehlo.subtract %arg248, %675 : tensor<1x8x8x512xf32>
      %677 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %678 = stablehlo.multiply %676, %677 : tensor<1x8x8x512xf32>
      %679 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %680 = stablehlo.multiply %678, %679 : tensor<1x8x8x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x8x8x512xf32>
      %682 = stablehlo.add %680, %681 : tensor<1x8x8x512xf32>
      mhlo.return %682 : tensor<1x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x8x8x512xf32>, tensor<512xf32>) -> tensor<1x8x8x512xf32>
    %670 = "mhlo.fusion"(%669, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x8x8x512xf32>, %arg246: tensor<f32>):
      %675 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x8x8x512xf32>, tensor<f32>) -> tensor<1x512xf32>
      mhlo.return %675 : tensor<1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x8x512xf32>, tensor<f32>) -> tensor<1x512xf32>
    %671 = "mhlo.fusion"(%670) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512xf32>):
      %cst_1 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %675 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x512xf32>
      %676 = stablehlo.multiply %arg245, %675 : tensor<1x512xf32>
      %677 = mhlo.bitcast %676 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x512xf32>) -> tensor<512xf32>
      mhlo.return %677 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512xf32>) -> tensor<512xf32>
    %672 = "mhlo.fusion"(%arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<2x512xf32>):
      %675 = stablehlo.transpose %arg245, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,2]{0,1}"} : (tensor<2x512xf32>) -> tensor<512x2xf32>
      %676 = mhlo.copy %675 : tensor<512x2xf32>
      mhlo.return %676 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<2x512xf32>) -> tensor<512x2xf32>
    %673 = "mhlo.fusion"(%arg244, %671, %672) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg245: tensor<2xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512x2xf32>):
      %675 = stablehlo.dot %arg246, %arg247, precision = [DEFAULT, DEFAULT] : (tensor<512xf32>, tensor<512x2xf32>) -> tensor<2xf32>
      %676 = stablehlo.add %arg245, %675 : tensor<2xf32>
      mhlo.return %676 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2xf32>, tensor<512xf32>, tensor<512x2xf32>) -> tensor<2xf32>
    %674 = mhlo.bitcast %673 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2xf32>) -> tensor<1x2xf32>
    return %674 : tensor<1x2xf32>
  }
}
