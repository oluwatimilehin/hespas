module {
  func.func @main(%arg0: tensor<384xf32>, %arg1: tensor<768xf32>, %arg2: tensor<384xf32>, %arg3: tensor<16x49xf32>, %arg4: tensor<16x49xi32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<512xf32>, %arg8: tensor<256xf32>, %arg9: tensor<8x196xf32>, %arg10: tensor<49x196xi32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<4x196xf32>, %arg14: tensor<196x196xi32>, %arg15: tensor<256xf32>, %arg16: tensor<128xf32>, %arg17: tensor<64xf32>, %arg18: tensor<32xf32>, %arg19: tensor<16xf32>, %arg20: tensor<1x3x224x224xf32>, %arg21: tensor<16x3x3x3xf32>, %arg22: tensor<16xf32>, %arg23: tensor<16xf32>, %arg24: tensor<16xf32>, %arg25: tensor<32x16x3x3xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32xf32>, %arg29: tensor<64x32x3x3xf32>, %arg30: tensor<64xf32>, %arg31: tensor<64xf32>, %arg32: tensor<64xf32>, %arg33: tensor<128x64x3x3xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<256x128xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<128x128xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<256xf32>, %arg47: tensor<256x128xf32>, %arg48: tensor<256xf32>, %arg49: tensor<256xf32>, %arg50: tensor<256xf32>, %arg51: tensor<128x256xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<4x196xf32>, %arg57: tensor<196x196xi32>, %arg58: tensor<256xf32>, %arg59: tensor<256x128xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<128x128xf32>, %arg64: tensor<128xf32>, %arg65: tensor<128xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256x128xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<256xf32>, %arg73: tensor<128x256xf32>, %arg74: tensor<128xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128x128xf32>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<128xf32>, %arg81: tensor<640xf32>, %arg82: tensor<640x128xf32>, %arg83: tensor<640xf32>, %arg84: tensor<640xf32>, %arg85: tensor<640xf32>, %arg86: tensor<256x512xf32>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<512x256xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<256x512xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<6x49xf32>, %arg100: tensor<49x49xi32>, %arg101: tensor<384xf32>, %arg102: tensor<384x256xf32>, %arg103: tensor<384xf32>, %arg104: tensor<384xf32>, %arg105: tensor<384xf32>, %arg106: tensor<256x192xf32>, %arg107: tensor<256xf32>, %arg108: tensor<256xf32>, %arg109: tensor<256xf32>, %arg110: tensor<256xf32>, %arg111: tensor<512xf32>, %arg112: tensor<512x256xf32>, %arg113: tensor<512xf32>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<256x512xf32>, %arg117: tensor<256xf32>, %arg118: tensor<256xf32>, %arg119: tensor<256xf32>, %arg120: tensor<256xf32>, %arg121: tensor<6x49xf32>, %arg122: tensor<49x49xi32>, %arg123: tensor<384xf32>, %arg124: tensor<384x256xf32>, %arg125: tensor<384xf32>, %arg126: tensor<384xf32>, %arg127: tensor<384xf32>, %arg128: tensor<256x192xf32>, %arg129: tensor<256xf32>, %arg130: tensor<256xf32>, %arg131: tensor<256xf32>, %arg132: tensor<256xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512x256xf32>, %arg135: tensor<512xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<256x512xf32>, %arg139: tensor<256xf32>, %arg140: tensor<256xf32>, %arg141: tensor<256xf32>, %arg142: tensor<256xf32>, %arg143: tensor<6x49xf32>, %arg144: tensor<49x49xi32>, %arg145: tensor<384xf32>, %arg146: tensor<384x256xf32>, %arg147: tensor<384xf32>, %arg148: tensor<384xf32>, %arg149: tensor<384xf32>, %arg150: tensor<256x192xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256xf32>, %arg155: tensor<512xf32>, %arg156: tensor<512x256xf32>, %arg157: tensor<512xf32>, %arg158: tensor<512xf32>, %arg159: tensor<512xf32>, %arg160: tensor<256x512xf32>, %arg161: tensor<256xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256x256xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<1280xf32>, %arg169: tensor<1280x256xf32>, %arg170: tensor<1280xf32>, %arg171: tensor<1280xf32>, %arg172: tensor<1280xf32>, %arg173: tensor<384x1024xf32>, %arg174: tensor<384xf32>, %arg175: tensor<384xf32>, %arg176: tensor<384xf32>, %arg177: tensor<768x384xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<384x768xf32>, %arg182: tensor<384xf32>, %arg183: tensor<384xf32>, %arg184: tensor<384xf32>, %arg185: tensor<384xf32>, %arg186: tensor<8x16xf32>, %arg187: tensor<16x16xi32>, %arg188: tensor<512xf32>, %arg189: tensor<512x384xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<512xf32>, %arg193: tensor<384x256xf32>, %arg194: tensor<384xf32>, %arg195: tensor<384xf32>, %arg196: tensor<384xf32>, %arg197: tensor<384xf32>, %arg198: tensor<768xf32>, %arg199: tensor<768x384xf32>, %arg200: tensor<768xf32>, %arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<384x768xf32>, %arg204: tensor<384xf32>, %arg205: tensor<384xf32>, %arg206: tensor<384xf32>, %arg207: tensor<384xf32>, %arg208: tensor<8x16xf32>, %arg209: tensor<16x16xi32>, %arg210: tensor<512xf32>, %arg211: tensor<512x384xf32>, %arg212: tensor<512xf32>, %arg213: tensor<512xf32>, %arg214: tensor<512xf32>, %arg215: tensor<384x256xf32>, %arg216: tensor<384xf32>, %arg217: tensor<384xf32>, %arg218: tensor<384xf32>, %arg219: tensor<384xf32>, %arg220: tensor<768xf32>, %arg221: tensor<768x384xf32>, %arg222: tensor<768xf32>, %arg223: tensor<768xf32>, %arg224: tensor<768xf32>, %arg225: tensor<384x768xf32>, %arg226: tensor<384xf32>, %arg227: tensor<384xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<8x16xf32>, %arg231: tensor<16x16xi32>, %arg232: tensor<512xf32>, %arg233: tensor<512x384xf32>, %arg234: tensor<512xf32>, %arg235: tensor<512xf32>, %arg236: tensor<512xf32>, %arg237: tensor<384x256xf32>, %arg238: tensor<384xf32>, %arg239: tensor<384xf32>, %arg240: tensor<384xf32>, %arg241: tensor<384xf32>, %arg242: tensor<768xf32>, %arg243: tensor<768x384xf32>, %arg244: tensor<768xf32>, %arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<384x768xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>, %arg250: tensor<384xf32>, %arg251: tensor<384xf32>, %arg252: tensor<8x16xf32>, %arg253: tensor<16x16xi32>, %arg254: tensor<512xf32>, %arg255: tensor<512x384xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<512xf32>, %arg259: tensor<384x256xf32>, %arg260: tensor<384xf32>, %arg261: tensor<384xf32>, %arg262: tensor<384xf32>, %arg263: tensor<384xf32>, %arg264: tensor<768xf32>, %arg265: tensor<768x384xf32>, %arg266: tensor<768xf32>, %arg267: tensor<768xf32>, %arg268: tensor<768xf32>, %arg269: tensor<384x768xf32>, %arg270: tensor<384xf32>, %arg271: tensor<384xf32>, %arg272: tensor<384xf32>, %arg273: tensor<384xf32>, %arg274: tensor<384xf32>, %arg275: tensor<384xf32>, %arg276: tensor<384xf32>, %arg277: tensor<1000x384xf32>, %arg278: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %2 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %3 = "mhlo.fusion"(%arg3, %arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x49xf32>, %arg280: tensor<16x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x49xi32>, tensor<16x49xi32>) -> tensor<16x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x49xi1>, tensor<16x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[784,1]{0,1}"} : (tensor<16x49xi32>) -> tensor<784x1xi32>
      %295 = mhlo.copy %294 : tensor<784x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 1>}> : (tensor<16x49xf32>, tensor<784x1xi32>) -> tensor<784x16x1xf32>
      mhlo.return %296 : tensor<784x16x1xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49xf32>, tensor<16x49xi32>) -> tensor<784x16x1xf32>
    %4 = "mhlo.fusion"(%arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %5 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %6 = "mhlo.fusion"(%arg7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %7 = "mhlo.fusion"(%arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %8 = "mhlo.fusion"(%arg9, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x196xf32>, %arg280: tensor<49x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x196xi32>, tensor<49x196xi32>) -> tensor<49x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x196xi1>, tensor<49x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[9604,1]{0,1}"} : (tensor<49x196xi32>) -> tensor<9604x1xi32>
      %295 = mhlo.copy %294 : tensor<9604x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x196xf32>, tensor<9604x1xi32>) -> tensor<9604x8x1xf32>
      mhlo.return %296 : tensor<9604x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x196xf32>, tensor<49x196xi32>) -> tensor<9604x8x1xf32>
    %9 = "mhlo.fusion"(%arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %10 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = "mhlo.fusion"(%arg13, %arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196xf32>, %arg280: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<196x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<196x196xi1>, tensor<196x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %295 = mhlo.copy %294 : tensor<38416x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      mhlo.return %296 : tensor<38416x4x1xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<38416x4x1xf32>
    %12 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %14 = "mhlo.fusion"(%arg17) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<64xf32>
      %291 = stablehlo.rsqrt %290 : tensor<64xf32>
      mhlo.return %291 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %15 = "mhlo.fusion"(%arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<32xf32>
      %291 = stablehlo.rsqrt %290 : tensor<32xf32>
      mhlo.return %291 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %16 = "mhlo.fusion"(%arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<16xf32>
      %291 = stablehlo.rsqrt %290 : tensor<16xf32>
      mhlo.return %291 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %17 = "mhlo.fusion"(%arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x3x224x224xf32>):
      %289 = stablehlo.transpose %arg279, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %290 = mhlo.copy %289 : tensor<1x224x224x3xf32>
      mhlo.return %290 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %18 = "mhlo.fusion"(%arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x3x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,16]{1,0,2,3}"} : (tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32>
      %290 = mhlo.copy %289 : tensor<3x3x3x16xf32>
      mhlo.return %290 : tensor<3x3x3x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x3x3x3xf32>) -> tensor<3x3x3x16xf32>
    %19 = stablehlo.convolution(%17, %18) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x112x112x16xf32>
    %20 = "mhlo.fusion"(%arg22, %arg23, %16, %19, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16xf32>, %arg280: tensor<16xf32>, %arg281: tensor<16xf32>, %arg282: tensor<1x112x112x16xf32>, %arg283: tensor<16xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x112x112x16xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x112x112x16xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x112x112x16xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<16xf32>) -> tensor<1x112x112x16xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x112x112x16xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,112,112]{1,3,2,0}"} : (tensor<1x112x112x16xf32>) -> tensor<1x16x112x112xf32>
      %298 = mhlo.copy %297 : tensor<1x16x112x112xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x16x112x112xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x16x112x112xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x16x112x112xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x112x112xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x16x112x112xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,112,112,16]{2,1,3,0}"} : (tensor<1x16x112x112xf32>) -> tensor<1x112x112x16xf32>
      %308 = mhlo.copy %307 : tensor<1x112x112x16xf32>
      mhlo.return %308 : tensor<1x112x112x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<1x112x112x16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %21 = "mhlo.fusion"(%arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32x16x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,16,32]{1,0,2,3}"} : (tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32>
      %290 = mhlo.copy %289 : tensor<3x3x16x32xf32>
      mhlo.return %290 : tensor<3x3x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x16x3x3xf32>) -> tensor<3x3x16x32xf32>
    %22 = stablehlo.convolution(%20, %21) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x112x112x16xf32>, tensor<3x3x16x32xf32>) -> tensor<1x56x56x32xf32>
    %23 = "mhlo.fusion"(%arg26, %arg27, %15, %22, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<32xf32>, %arg280: tensor<32xf32>, %arg281: tensor<32xf32>, %arg282: tensor<1x56x56x32xf32>, %arg283: tensor<32xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x56x56x32xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x56x56x32xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56x32xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<32xf32>) -> tensor<1x56x56x32xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x56x56x32xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,56,56]{1,3,2,0}"} : (tensor<1x56x56x32xf32>) -> tensor<1x32x56x56xf32>
      %298 = mhlo.copy %297 : tensor<1x32x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x32x56x56xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x32x56x56xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x32x56x56xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x32x56x56xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x32x56x56xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,56,56,32]{2,1,3,0}"} : (tensor<1x32x56x56xf32>) -> tensor<1x56x56x32xf32>
      %308 = mhlo.copy %307 : tensor<1x56x56x32xf32>
      mhlo.return %308 : tensor<1x56x56x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1x56x56x32xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
    %24 = "mhlo.fusion"(%arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64x32x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %290 = mhlo.copy %289 : tensor<3x3x32x64xf32>
      mhlo.return %290 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x32xf32>, tensor<3x3x32x64xf32>) -> tensor<1x28x28x64xf32>
    %26 = "mhlo.fusion"(%arg30, %arg31, %14, %25, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<64xf32>, %arg280: tensor<64xf32>, %arg281: tensor<64xf32>, %arg282: tensor<1x28x28x64xf32>, %arg283: tensor<64xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x28x28x64xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x28x28x64xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28x64xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<64xf32>) -> tensor<1x28x28x64xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x28x28x64xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,64,28,28]{1,3,2,0}"} : (tensor<1x28x28x64xf32>) -> tensor<1x64x28x28xf32>
      %298 = mhlo.copy %297 : tensor<1x64x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %301 = stablehlo.add %298, %300 : tensor<1x64x28x28xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %303 = stablehlo.clamp %299, %301, %302 : tensor<1x64x28x28xf32>
      %304 = stablehlo.multiply %298, %303 : tensor<1x64x28x28xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x64x28x28xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x64x28x28xf32>
      %307 = stablehlo.transpose %306, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,28,28,64]{2,1,3,0}"} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
      %308 = mhlo.copy %307 : tensor<1x28x28x64xf32>
      mhlo.return %308 : tensor<1x28x28x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x28x28x64xf32>, tensor<64xf32>) -> tensor<1x28x28x64xf32>
    %27 = "mhlo.fusion"(%arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128x64x3x3xf32>):
      %289 = stablehlo.transpose %arg279, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,128]{1,0,2,3}"} : (tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32>
      %290 = mhlo.copy %289 : tensor<3x3x64x128xf32>
      mhlo.return %290 : tensor<3x3x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x3x3xf32>) -> tensor<3x3x64x128xf32>
    %28 = stablehlo.convolution(%26, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x14x14x128xf32>
    %29 = "mhlo.fusion"(%arg34, %arg35, %13, %28, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<1x14x14x128xf32>, %arg283: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<1x14x14x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x14x14x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14x128xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
      %298 = mhlo.copy %297 : tensor<1x128x14x14xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
      %301 = mhlo.copy %300 : tensor<1x196x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %302 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x14x14x128xf32>, tensor<128xf32>) -> tensor<196x128xf32>
    %30 = stablehlo.dot_general %29, %arg37, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %31 = "mhlo.fusion"(%arg38, %arg39, %12, %30, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
      mhlo.return %301 : tensor<4x196x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x16xf32>
    %32 = "mhlo.fusion"(%arg38, %arg39, %12, %30, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x4x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
      mhlo.return %301 : tensor<4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x16x196xf32>
    %33 = stablehlo.dot_general %31, %32, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x16xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %34 = "mhlo.fusion"(%11, %33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<38416x4x1xf32>, %arg280: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      mhlo.return %295 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
      mhlo.return %289 : tensor<1x4x196x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    %36 = "mhlo.fusion"(%35, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %37 = "mhlo.fusion"(%36, %11, %33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196xf32>, %arg280: tensor<38416x4x1xf32>, %arg281: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x4x196xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x4x196x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x4x196x196xf32>
      mhlo.return %301 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196xf32>, tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = "mhlo.fusion"(%37, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %39 = "mhlo.fusion"(%37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<1x4x196xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x4x196x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
      mhlo.return %292 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<1x4x196xf32>) -> tensor<4x196x196xf32>
    %40 = "mhlo.fusion"(%arg38, %arg39, %12, %30, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
      mhlo.return %301 : tensor<4x196x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x32xf32>
    %41 = stablehlo.dot_general %39, %40, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x196x32xf32>) -> tensor<4x196x32xf32>
    %42 = "mhlo.fusion"(%41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
      %291 = mhlo.copy %290 : tensor<1x196x4x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x196x128xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x196x128xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x196x128xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x196x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %301 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x32xf32>) -> tensor<196x128xf32>
    %43 = stablehlo.dot_general %42, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %44 = "mhlo.fusion"(%arg42, %arg43, %10, %43, %arg44, %arg34, %arg35, %13, %28, %arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<196x128xf32>, %arg283: tensor<128xf32>, %arg284: tensor<128xf32>, %arg285: tensor<128xf32>, %arg286: tensor<128xf32>, %arg287: tensor<1x14x14x128xf32>, %arg288: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<1x14x14x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<1x14x14x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<128xf32>) -> tensor<1x14x14x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14x128xf32>
      %297 = stablehlo.transpose %296, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,14,14]{1,3,2,0}"} : (tensor<1x14x14x128xf32>) -> tensor<1x128x14x14xf32>
      %298 = mhlo.copy %297 : tensor<1x128x14x14xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x128x14x14xf32>) -> tensor<1x128x196xf32>
      %300 = stablehlo.transpose %299, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,128]{1,2,0}"} : (tensor<1x128x196xf32>) -> tensor<1x196x128xf32>
      %301 = mhlo.copy %300 : tensor<1x196x128xf32>
      %302 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %303 = stablehlo.subtract %arg282, %302 : tensor<196x128xf32>
      %304 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<196x128xf32>
      %306 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<196x128xf32>
      %308 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %309 = stablehlo.add %307, %308 : tensor<196x128xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %311 = stablehlo.add %301, %310 : tensor<1x196x128xf32>
      mhlo.return %311 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x14x14x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %45 = "mhlo.fusion"(%arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %46 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %47 = mhlo.bitcast %44 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %48 = stablehlo.dot_general %47, %arg47, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %49 = "mhlo.fusion"(%arg48, %arg49, %46, %48, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x196x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x196x256xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x196x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x196x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
      mhlo.return %306 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<196x256xf32>
    %50 = stablehlo.dot_general %49, %arg51, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %51 = "mhlo.fusion"(%44, %arg52, %arg53, %45, %50, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      mhlo.return %298 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %52 = "mhlo.fusion"(%arg55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %53 = "mhlo.fusion"(%arg56, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196xf32>, %arg280: tensor<196x196xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<196x196xi32>, tensor<196x196xi32>) -> tensor<196x196xi1>
      %c_1 = stablehlo.constant dense<196> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<196x196xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<196x196xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<196x196xi1>, tensor<196x196xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[38416,1]{0,1}"} : (tensor<196x196xi32>) -> tensor<38416x1xi32>
      %295 = mhlo.copy %294 : tensor<38416x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1>}> : (tensor<4x196xf32>, tensor<38416x1xi32>) -> tensor<38416x4x1xf32>
      mhlo.return %296 : tensor<38416x4x1xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196xf32>, tensor<196x196xi32>) -> tensor<38416x4x1xf32>
    %54 = "mhlo.fusion"(%arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %55 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %56 = stablehlo.dot_general %55, %arg59, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %57 = "mhlo.fusion"(%arg60, %arg61, %54, %56, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 0:16] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,16]{3,1,2,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x196x16xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x16xf32>) -> tensor<4x196x16xf32>
      mhlo.return %301 : tensor<4x196x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x16xf32>
    %58 = "mhlo.fusion"(%arg60, %arg61, %54, %56, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 16:32] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,16,196]{2,1,3,0}"} : (tensor<1x196x4x16xf32>) -> tensor<1x4x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x4x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x16x196xf32>) -> tensor<4x16x196xf32>
      mhlo.return %301 : tensor<4x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x16x196xf32>
    %59 = stablehlo.dot_general %57, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x16xf32>, tensor<4x16x196xf32>) -> tensor<4x196x196xf32>
    %60 = "mhlo.fusion"(%53, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<38416x4x1xf32>, %arg280: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      mhlo.return %295 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %61 = "mhlo.fusion"(%60, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
      mhlo.return %289 : tensor<1x4x196x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196x7xf32>
    %62 = "mhlo.fusion"(%61, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x7xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %63 = "mhlo.fusion"(%62, %53, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196xf32>, %arg280: tensor<38416x4x1xf32>, %arg281: tensor<4x196x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x196x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<4x196x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<38416x4x1xf32>) -> tensor<196x196x4xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,196,196]{0,2,1}"} : (tensor<196x196x4xf32>) -> tensor<4x196x196xf32>
      %293 = mhlo.copy %292 : tensor<4x196x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<4x196x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x196xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x4x196xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x4x196x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x4x196x196xf32>
      mhlo.return %301 : tensor<1x4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196xf32>, tensor<38416x4x1xf32>, tensor<4x196x196xf32>) -> tensor<1x4x196x196xf32>
    %64 = "mhlo.fusion"(%63, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
      mhlo.return %289 : tensor<1x4x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<f32>) -> tensor<1x4x196xf32>
    %65 = "mhlo.fusion"(%63, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x4x196x196xf32>, %arg280: tensor<1x4x196xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x4x196xf32>) -> tensor<4x196xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<4x196xf32>) -> tensor<1x4x196x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x4x196x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x196xf32>) -> tensor<4x196x196xf32>
      mhlo.return %292 : tensor<4x196x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x4x196x196xf32>, tensor<1x4x196xf32>) -> tensor<4x196x196xf32>
    %66 = "mhlo.fusion"(%arg60, %arg61, %54, %56, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x4x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:4, 32:64] : (tensor<1x196x4x64xf32>) -> tensor<1x196x4x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,196,32]{3,1,2,0}"} : (tensor<1x196x4x32xf32>) -> tensor<1x4x196x32xf32>
      %300 = mhlo.copy %299 : tensor<1x4x196x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x196x32xf32>) -> tensor<4x196x32xf32>
      mhlo.return %301 : tensor<4x196x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<4x196x32xf32>
    %67 = stablehlo.dot_general %65, %66, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x196x196xf32>, tensor<4x196x32xf32>) -> tensor<4x196x32xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<4x196x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<4x196x32xf32>) -> tensor<1x4x196x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,196,4,32]{3,1,2,0}"} : (tensor<1x4x196x32xf32>) -> tensor<1x196x4x32xf32>
      %291 = mhlo.copy %290 : tensor<1x196x4x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x196x4x32xf32>) -> tensor<1x196x128xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x196x128xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x196x128xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x196x128xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x128xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x196x128xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %301 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<4x196x32xf32>) -> tensor<196x128xf32>
    %69 = stablehlo.dot_general %68, %arg63, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<128x128xf32>) -> tensor<196x128xf32>
    %70 = "mhlo.fusion"(%51, %arg64, %arg65, %52, %69, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      mhlo.return %298 : tensor<1x196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<1x196x128xf32>
    %71 = "mhlo.fusion"(%arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<128xf32>
      %291 = stablehlo.rsqrt %290 : tensor<128xf32>
      mhlo.return %291 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %72 = "mhlo.fusion"(%arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %73 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
    %74 = stablehlo.dot_general %73, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<256x128xf32>) -> tensor<196x256xf32>
    %75 = "mhlo.fusion"(%arg70, %arg71, %72, %74, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<196x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<196x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x256xf32>) -> tensor<1x196x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x196x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x196x256xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x196x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x196x256xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x196x256xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x256xf32>) -> tensor<196x256xf32>
      mhlo.return %306 : tensor<196x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<196x256xf32>, tensor<256xf32>) -> tensor<196x256xf32>
    %76 = stablehlo.dot_general %75, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x256xf32>, tensor<128x256xf32>) -> tensor<196x128xf32>
    %77 = "mhlo.fusion"(%70, %arg74, %arg75, %71, %76, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<1x14x14x128xf32>
      %300 = stablehlo.slice %299 [0:1, 0:14:2, 0:14, 0:128] : (tensor<1x14x14x128xf32>) -> tensor<1x7x14x128xf32>
      %301 = stablehlo.slice %300 [0:1, 0:7, 0:14:2, 0:128] : (tensor<1x7x14x128xf32>) -> tensor<1x7x7x128xf32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x128xf32>) -> tensor<49x128xf32>
      mhlo.return %302 : tensor<49x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<49x128xf32>
    %78 = stablehlo.dot_general %77, %arg77, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x128xf32>, tensor<128x128xf32>) -> tensor<49x128xf32>
    %79 = "mhlo.fusion"(%arg78, %arg79, %9, %78, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<49x128xf32>, %arg283: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<128xf32>) -> tensor<49x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x128xf32>) -> tensor<1x49x8x16xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,49,16]{3,1,2,0}"} : (tensor<1x49x8x16xf32>) -> tensor<1x8x49x16xf32>
      %299 = mhlo.copy %298 : tensor<1x8x49x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x16xf32>) -> tensor<8x49x16xf32>
      mhlo.return %300 : tensor<8x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<49x128xf32>, tensor<128xf32>) -> tensor<8x49x16xf32>
    %80 = "mhlo.fusion"(%arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<640xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<640xf32>
      %291 = stablehlo.rsqrt %290 : tensor<640xf32>
      mhlo.return %291 : tensor<640xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>) -> tensor<640xf32>
    %81 = "mhlo.fusion"(%70, %arg74, %arg75, %71, %76, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x196x128xf32>, %arg280: tensor<128xf32>, %arg281: tensor<128xf32>, %arg282: tensor<128xf32>, %arg283: tensor<196x128xf32>, %arg284: tensor<128xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<196x128xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x128xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<128xf32>) -> tensor<196x128xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x128xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x128xf32>) -> tensor<1x196x128xf32>
      %298 = stablehlo.add %arg279, %297 : tensor<1x196x128xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x196x128xf32>) -> tensor<196x128xf32>
      mhlo.return %299 : tensor<196x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x196x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<196x128xf32>, tensor<128xf32>) -> tensor<196x128xf32>
    %82 = stablehlo.dot_general %81, %arg82, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x128xf32>, tensor<640x128xf32>) -> tensor<196x640xf32>
    %83 = "mhlo.fusion"(%arg83, %arg84, %80, %82, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<196x640xf32>, %arg283: tensor<640xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x640xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x640xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x640xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x640xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:8, 0:16] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,196]{2,1,3,0}"} : (tensor<1x196x8x16xf32>) -> tensor<1x8x16x196xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x196xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x196xf32>) -> tensor<8x16x196xf32>
      mhlo.return %301 : tensor<8x16x196xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<196x640xf32>, tensor<640xf32>) -> tensor<8x16x196xf32>
    %84 = stablehlo.dot_general %79, %83, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x49x16xf32>, tensor<8x16x196xf32>) -> tensor<8x49x196xf32>
    %85 = "mhlo.fusion"(%8, %84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<9604x8x1xf32>, %arg280: tensor<8x49x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x49x196xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
      %293 = mhlo.copy %292 : tensor<8x49x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x49x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
      mhlo.return %295 : tensor<1x8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<9604x8x1xf32>, tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [14, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49x7xf32>
      mhlo.return %289 : tensor<1x8x49x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49x7xf32>
    %87 = "mhlo.fusion"(%86, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x7xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x49x7xf32>, tensor<f32>) -> tensor<1x8x49xf32>
      mhlo.return %289 : tensor<1x8x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x7xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    %88 = "mhlo.fusion"(%87, %8, %84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49xf32>, %arg280: tensor<9604x8x1xf32>, %arg281: tensor<8x49x196xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x49x196xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x49x196xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<9604x8x1xf32>) -> tensor<49x196x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,196]{0,2,1}"} : (tensor<49x196x8xf32>) -> tensor<8x49x196xf32>
      %293 = mhlo.copy %292 : tensor<8x49x196xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x49x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x49x196xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x49x196xf32>
      mhlo.return %301 : tensor<1x8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49xf32>, tensor<9604x8x1xf32>, tensor<8x49x196xf32>) -> tensor<1x8x49x196xf32>
    %89 = "mhlo.fusion"(%88, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49xf32>
      mhlo.return %289 : tensor<1x8x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<f32>) -> tensor<1x8x49xf32>
    %90 = "mhlo.fusion"(%88, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x49x196xf32>, %arg280: tensor<1x8x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49xf32>) -> tensor<8x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x49xf32>) -> tensor<1x8x49x196xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x49x196xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x49x196xf32>) -> tensor<8x49x196xf32>
      mhlo.return %292 : tensor<8x49x196xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x49x196xf32>, tensor<1x8x49xf32>) -> tensor<8x49x196xf32>
    %91 = "mhlo.fusion"(%arg83, %arg84, %80, %82, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<640xf32>, %arg280: tensor<640xf32>, %arg281: tensor<640xf32>, %arg282: tensor<196x640xf32>, %arg283: tensor<640xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<196x640xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<196x640xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<196x640xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<640xf32>) -> tensor<196x640xf32>
      %296 = stablehlo.add %294, %295 : tensor<196x640xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x640xf32>) -> tensor<1x196x8x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:196, 0:8, 16:80] : (tensor<1x196x8x80xf32>) -> tensor<1x196x8x64xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,196,64]{3,1,2,0}"} : (tensor<1x196x8x64xf32>) -> tensor<1x8x196x64xf32>
      %300 = mhlo.copy %299 : tensor<1x8x196x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x196x64xf32>) -> tensor<8x196x64xf32>
      mhlo.return %301 : tensor<8x196x64xf32>
    }) {output_operand_aliasing = []} : (tensor<640xf32>, tensor<640xf32>, tensor<640xf32>, tensor<196x640xf32>, tensor<640xf32>) -> tensor<8x196x64xf32>
    %92 = stablehlo.dot_general %90, %91, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x49x196xf32>, tensor<8x196x64xf32>) -> tensor<8x49x64xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x49x64xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x49x64xf32>) -> tensor<1x8x49x64xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,8,64]{3,1,2,0}"} : (tensor<1x8x49x64xf32>) -> tensor<1x49x8x64xf32>
      %291 = mhlo.copy %290 : tensor<1x49x8x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x8x64xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x512xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x512xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %301 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<8x49x64xf32>) -> tensor<49x512xf32>
    %94 = stablehlo.dot_general %93, %arg86, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %95 = "mhlo.fusion"(%arg87, %arg88, %7, %94, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<49x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      mhlo.return %296 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %96 = stablehlo.dot_general %95, %arg90, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %97 = "mhlo.fusion"(%arg91, %arg92, %6, %96, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %98 = stablehlo.dot_general %97, %arg94, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %99 = "mhlo.fusion"(%arg95, %arg96, %5, %98, %arg97, %arg87, %arg88, %7, %94, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<49x256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<49x256xf32>, %arg288: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %298 = stablehlo.subtract %arg282, %297 : tensor<49x256xf32>
      %299 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<49x256xf32>
      %301 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<49x256xf32>
      %303 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %304 = stablehlo.add %302, %303 : tensor<49x256xf32>
      %305 = stablehlo.add %296, %304 : tensor<49x256xf32>
      mhlo.return %305 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %100 = "mhlo.fusion"(%arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %101 = "mhlo.fusion"(%arg99, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %102 = "mhlo.fusion"(%arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %103 = stablehlo.dot_general %99, %arg102, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %104 = "mhlo.fusion"(%arg103, %arg104, %102, %103, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %105 = "mhlo.fusion"(%arg103, %arg104, %102, %103, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %106 = stablehlo.dot_general %104, %105, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %107 = "mhlo.fusion"(%101, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %110 = "mhlo.fusion"(%109, %101, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %111 = "mhlo.fusion"(%110, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %112 = "mhlo.fusion"(%111, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %113 = "mhlo.fusion"(%110, %112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %114 = "mhlo.fusion"(%arg103, %arg104, %102, %103, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %115 = stablehlo.dot_general %113, %114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %117 = stablehlo.dot_general %116, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %118 = "mhlo.fusion"(%99, %arg107, %arg108, %100, %117, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %119 = "mhlo.fusion"(%arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %120 = "mhlo.fusion"(%arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %121 = stablehlo.dot_general %118, %arg112, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %122 = "mhlo.fusion"(%arg113, %arg114, %120, %121, %arg115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %123 = stablehlo.dot_general %122, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %124 = "mhlo.fusion"(%118, %arg117, %arg118, %119, %123, %arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %125 = "mhlo.fusion"(%arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %126 = "mhlo.fusion"(%arg121, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %127 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %128 = stablehlo.dot_general %124, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %129 = "mhlo.fusion"(%arg125, %arg126, %127, %128, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %130 = "mhlo.fusion"(%arg125, %arg126, %127, %128, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %131 = stablehlo.dot_general %129, %130, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %132 = "mhlo.fusion"(%126, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %134 = "mhlo.fusion"(%133, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %135 = "mhlo.fusion"(%134, %126, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %136 = "mhlo.fusion"(%135, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %137 = "mhlo.fusion"(%136, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %138 = "mhlo.fusion"(%135, %137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %139 = "mhlo.fusion"(%arg125, %arg126, %127, %128, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %140 = stablehlo.dot_general %138, %139, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %141 = "mhlo.fusion"(%140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %142 = stablehlo.dot_general %141, %arg128, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %143 = "mhlo.fusion"(%124, %arg129, %arg130, %125, %142, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %144 = "mhlo.fusion"(%arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %145 = "mhlo.fusion"(%arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %146 = stablehlo.dot_general %143, %arg134, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %147 = "mhlo.fusion"(%arg135, %arg136, %145, %146, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %148 = stablehlo.dot_general %147, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %149 = "mhlo.fusion"(%143, %arg139, %arg140, %144, %148, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %150 = "mhlo.fusion"(%arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %151 = "mhlo.fusion"(%arg143, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49xf32>, %arg280: tensor<49x49xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<49x49xi32>, tensor<49x49xi32>) -> tensor<49x49xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<49x49xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<49x49xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<49x49xi1>, tensor<49x49xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2401,1]{0,1}"} : (tensor<49x49xi32>) -> tensor<2401x1xi32>
      %295 = mhlo.copy %294 : tensor<2401x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x49xf32>, tensor<2401x1xi32>) -> tensor<2401x6x1xf32>
      mhlo.return %296 : tensor<2401x6x1xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49xf32>, tensor<49x49xi32>) -> tensor<2401x6x1xf32>
    %152 = "mhlo.fusion"(%arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %153 = stablehlo.dot_general %149, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<384x256xf32>) -> tensor<49x384xf32>
    %154 = "mhlo.fusion"(%arg147, %arg148, %152, %153, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 0:16] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,16]{3,1,2,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x49x16xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x16xf32>) -> tensor<6x49x16xf32>
      mhlo.return %301 : tensor<6x49x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x16xf32>
    %155 = "mhlo.fusion"(%arg147, %arg148, %152, %153, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 16:32] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,16,49]{2,1,3,0}"} : (tensor<1x49x6x16xf32>) -> tensor<1x6x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x6x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x16x49xf32>) -> tensor<6x16x49xf32>
      mhlo.return %301 : tensor<6x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x16x49xf32>
    %156 = stablehlo.dot_general %154, %155, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x16xf32>, tensor<6x16x49xf32>) -> tensor<6x49x49xf32>
    %157 = "mhlo.fusion"(%151, %156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<2401x6x1xf32>, %arg280: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      mhlo.return %295 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %158 = "mhlo.fusion"(%157, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %159 = "mhlo.fusion"(%158, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %160 = "mhlo.fusion"(%159, %151, %156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49xf32>, %arg280: tensor<2401x6x1xf32>, %arg281: tensor<6x49x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x49x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<6x49x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<2401x6x1xf32>) -> tensor<49x49x6xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[6,49,49]{0,2,1}"} : (tensor<49x49x6xf32>) -> tensor<6x49x49xf32>
      %293 = mhlo.copy %292 : tensor<6x49x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<6x49x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x6x49xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x6x49xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x6x49x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x6x49x49xf32>
      mhlo.return %301 : tensor<1x6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49xf32>, tensor<2401x6x1xf32>, tensor<6x49x49xf32>) -> tensor<1x6x49x49xf32>
    %161 = "mhlo.fusion"(%160, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
      mhlo.return %289 : tensor<1x6x49x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<f32>) -> tensor<1x6x49x2xf32>
    %162 = "mhlo.fusion"(%161, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
      mhlo.return %289 : tensor<1x6x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x2xf32>, tensor<f32>) -> tensor<1x6x49xf32>
    %163 = "mhlo.fusion"(%160, %162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x6x49x49xf32>, %arg280: tensor<1x6x49xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x6x49xf32>) -> tensor<6x49xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<6x49xf32>) -> tensor<1x6x49x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x6x49x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x49xf32>) -> tensor<6x49x49xf32>
      mhlo.return %292 : tensor<6x49x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x6x49x49xf32>, tensor<1x6x49xf32>) -> tensor<6x49x49xf32>
    %164 = "mhlo.fusion"(%arg147, %arg148, %152, %153, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<49x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<49x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x384xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x384xf32>) -> tensor<1x49x6x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:6, 32:64] : (tensor<1x49x6x64xf32>) -> tensor<1x49x6x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,6,49,32]{3,1,2,0}"} : (tensor<1x49x6x32xf32>) -> tensor<1x6x49x32xf32>
      %300 = mhlo.copy %299 : tensor<1x6x49x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x6x49x32xf32>) -> tensor<6x49x32xf32>
      mhlo.return %301 : tensor<6x49x32xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<49x384xf32>, tensor<384xf32>) -> tensor<6x49x32xf32>
    %165 = stablehlo.dot_general %163, %164, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6x49x49xf32>, tensor<6x49x32xf32>) -> tensor<6x49x32xf32>
    %166 = "mhlo.fusion"(%165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<6x49x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<6x49x32xf32>) -> tensor<1x6x49x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,49,6,32]{3,1,2,0}"} : (tensor<1x6x49x32xf32>) -> tensor<1x49x6x32xf32>
      %291 = mhlo.copy %290 : tensor<1x49x6x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x49x6x32xf32>) -> tensor<1x49x192xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x49x192xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x49x192xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x49x192xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x192xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x49x192xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x192xf32>) -> tensor<49x192xf32>
      mhlo.return %301 : tensor<49x192xf32>
    }) {output_operand_aliasing = []} : (tensor<6x49x32xf32>) -> tensor<49x192xf32>
    %167 = stablehlo.dot_general %166, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x192xf32>, tensor<256x192xf32>) -> tensor<49x256xf32>
    %168 = "mhlo.fusion"(%149, %arg151, %arg152, %150, %167, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %169 = "mhlo.fusion"(%arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<256xf32>
      %291 = stablehlo.rsqrt %290 : tensor<256xf32>
      mhlo.return %291 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %170 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %171 = stablehlo.dot_general %168, %arg156, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<512x256xf32>) -> tensor<49x512xf32>
    %172 = "mhlo.fusion"(%arg157, %arg158, %170, %171, %arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<49x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<49x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x512xf32>) -> tensor<1x49x512xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x49x512xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x49x512xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x49x512xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x49x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x49x512xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x49x512xf32>) -> tensor<49x512xf32>
      mhlo.return %306 : tensor<49x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<49x512xf32>, tensor<512xf32>) -> tensor<49x512xf32>
    %173 = stablehlo.dot_general %172, %arg160, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x512xf32>, tensor<256x512xf32>) -> tensor<49x256xf32>
    %174 = "mhlo.fusion"(%168, %arg161, %arg162, %169, %173, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<49x256xf32>, %arg284: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<49x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<49x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x256xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<49x256xf32>
      mhlo.return %297 : tensor<49x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<49x256xf32>, tensor<256xf32>) -> tensor<49x256xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<49x256xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x256xf32>) -> tensor<1x7x7x256xf32>
      %290 = stablehlo.slice %289 [0:1, 0:7:2, 0:7, 0:256] : (tensor<1x7x7x256xf32>) -> tensor<1x4x7x256xf32>
      %291 = stablehlo.slice %290 [0:1, 0:4, 0:7:2, 0:256] : (tensor<1x4x7x256xf32>) -> tensor<1x4x4x256xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x4x4x256xf32>) -> tensor<16x256xf32>
      mhlo.return %292 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<49x256xf32>) -> tensor<16x256xf32>
    %176 = stablehlo.dot_general %175, %arg164, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<256x256xf32>) -> tensor<16x256xf32>
    %177 = "mhlo.fusion"(%arg165, %arg166, %4, %176, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<16x256xf32>, %arg283: tensor<256xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x256xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x256xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x256xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<256xf32>) -> tensor<16x256xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x256xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x256xf32>) -> tensor<1x16x16x16xf32>
      %298 = stablehlo.transpose %297, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,16]{3,1,2,0}"} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
      %299 = mhlo.copy %298 : tensor<1x16x16x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x16xf32>) -> tensor<16x16x16xf32>
      mhlo.return %300 : tensor<16x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256xf32>, tensor<256xf32>) -> tensor<16x16x16xf32>
    %178 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1280xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<1280xf32>
      %291 = stablehlo.rsqrt %290 : tensor<1280xf32>
      mhlo.return %291 : tensor<1280xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>) -> tensor<1280xf32>
    %179 = stablehlo.dot_general %174, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x256xf32>, tensor<1280x256xf32>) -> tensor<49x1280xf32>
    %180 = "mhlo.fusion"(%arg170, %arg171, %178, %179, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<49x1280xf32>, %arg283: tensor<1280xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x1280xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x1280xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x1280xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x1280xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:16, 0:16] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,49]{2,1,3,0}"} : (tensor<1x49x16x16xf32>) -> tensor<1x16x16x49xf32>
      %300 = mhlo.copy %299 : tensor<1x16x16x49xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
      mhlo.return %301 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<49x1280xf32>, tensor<1280xf32>) -> tensor<16x16x49xf32>
    %181 = stablehlo.dot_general %177, %180, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x16xf32>, tensor<16x16x49xf32>) -> tensor<16x16x49xf32>
    %182 = "mhlo.fusion"(%3, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<784x16x1xf32>, %arg280: tensor<16x16x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<16x16x49xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
      %293 = mhlo.copy %292 : tensor<16x16x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<16x16x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
      mhlo.return %295 : tensor<1x16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<784x16x1xf32>, tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    %183 = "mhlo.fusion"(%182, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.maximum %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
      mhlo.return %289 : tensor<1x16x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %289 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %185 = "mhlo.fusion"(%184, %3, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16xf32>, %arg280: tensor<784x16x1xf32>, %arg281: tensor<16x16x49xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x16x49xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<16x16x49xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<784x16x1xf32>) -> tensor<16x49x16xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,16,49]{0,2,1}"} : (tensor<16x49x16xf32>) -> tensor<16x16x49xf32>
      %293 = mhlo.copy %292 : tensor<16x16x49xf32>
      %294 = stablehlo.add %290, %293 : tensor<16x16x49xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x16x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x16x16x49xf32>
      %301 = stablehlo.exponential %300 : tensor<1x16x16x49xf32>
      mhlo.return %301 : tensor<1x16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16xf32>, tensor<784x16x1xf32>, tensor<16x16x49xf32>) -> tensor<1x16x16x49xf32>
    %186 = "mhlo.fusion"(%185, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<f32>):
      %289 = "stablehlo.reduce_window"(%arg279, %arg280) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg281: tensor<f32>, %arg282: tensor<f32>):
        %290 = stablehlo.add %arg281, %arg282 : tensor<f32>
        stablehlo.return %290 : tensor<f32>
      }) : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
      mhlo.return %289 : tensor<1x16x16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<f32>) -> tensor<1x16x16x2xf32>
    %187 = "mhlo.fusion"(%186, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x2xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
      mhlo.return %289 : tensor<1x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x2xf32>, tensor<f32>) -> tensor<1x16x16xf32>
    %188 = "mhlo.fusion"(%185, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x16x16x49xf32>, %arg280: tensor<1x16x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16x49xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x16x16x49xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x49xf32>) -> tensor<16x16x49xf32>
      mhlo.return %292 : tensor<16x16x49xf32>
    }) {output_operand_aliasing = []} : (tensor<1x16x16x49xf32>, tensor<1x16x16xf32>) -> tensor<16x16x49xf32>
    %189 = "mhlo.fusion"(%arg170, %arg171, %178, %179, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1280xf32>, %arg280: tensor<1280xf32>, %arg281: tensor<1280xf32>, %arg282: tensor<49x1280xf32>, %arg283: tensor<1280xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<49x1280xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<49x1280xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<49x1280xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<1280xf32>) -> tensor<49x1280xf32>
      %296 = stablehlo.add %294, %295 : tensor<49x1280xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x1280xf32>) -> tensor<1x49x16x80xf32>
      %298 = stablehlo.slice %297 [0:1, 0:49, 0:16, 16:80] : (tensor<1x49x16x80xf32>) -> tensor<1x49x16x64xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,49,64]{3,1,2,0}"} : (tensor<1x49x16x64xf32>) -> tensor<1x16x49x64xf32>
      %300 = mhlo.copy %299 : tensor<1x16x49x64xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x49x64xf32>) -> tensor<16x49x64xf32>
      mhlo.return %301 : tensor<16x49x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<49x1280xf32>, tensor<1280xf32>) -> tensor<16x49x64xf32>
    %190 = stablehlo.dot_general %188, %189, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x16x49xf32>, tensor<16x49x64xf32>) -> tensor<16x16x64xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x16x64xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x16x64xf32>) -> tensor<1x16x16x64xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,16,64]{3,1,2,0}"} : (tensor<1x16x16x64xf32>) -> tensor<1x16x16x64xf32>
      %291 = mhlo.copy %290 : tensor<1x16x16x64xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x16x64xf32>) -> tensor<1x16x1024xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x1024xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x1024xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x1024xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x1024xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x1024xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x1024xf32>) -> tensor<16x1024xf32>
      mhlo.return %301 : tensor<16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16x16x64xf32>) -> tensor<16x1024xf32>
    %192 = stablehlo.dot_general %191, %arg173, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x1024xf32>, tensor<384x1024xf32>) -> tensor<16x384xf32>
    %193 = "mhlo.fusion"(%arg174, %arg175, %2, %192, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<16x384xf32>, %arg283: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      mhlo.return %296 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %194 = stablehlo.dot_general %193, %arg177, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %195 = "mhlo.fusion"(%arg178, %arg179, %1, %194, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %196 = stablehlo.dot_general %195, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %197 = "mhlo.fusion"(%arg182, %arg183, %0, %196, %arg184, %arg174, %arg175, %2, %192, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<16x384xf32>, %arg283: tensor<384xf32>, %arg284: tensor<384xf32>, %arg285: tensor<384xf32>, %arg286: tensor<384xf32>, %arg287: tensor<16x384xf32>, %arg288: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %298 = stablehlo.subtract %arg282, %297 : tensor<16x384xf32>
      %299 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<16x384xf32>
      %301 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<16x384xf32>
      %303 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %304 = stablehlo.add %302, %303 : tensor<16x384xf32>
      %305 = stablehlo.add %296, %304 : tensor<16x384xf32>
      mhlo.return %305 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %198 = "mhlo.fusion"(%arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %199 = "mhlo.fusion"(%arg186, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %200 = "mhlo.fusion"(%arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %201 = stablehlo.dot_general %197, %arg189, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %202 = "mhlo.fusion"(%arg190, %arg191, %200, %201, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %203 = "mhlo.fusion"(%arg190, %arg191, %200, %201, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %204 = stablehlo.dot_general %202, %203, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %205 = "mhlo.fusion"(%199, %204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %206 = "mhlo.fusion"(%205, %199, %204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %207 = "mhlo.fusion"(%206, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %208 = "mhlo.fusion"(%206, %207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %209 = "mhlo.fusion"(%arg190, %arg191, %200, %201, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %210 = stablehlo.dot_general %208, %209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %212 = stablehlo.dot_general %211, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %213 = "mhlo.fusion"(%197, %arg194, %arg195, %198, %212, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %214 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %215 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %216 = stablehlo.dot_general %213, %arg199, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %217 = "mhlo.fusion"(%arg200, %arg201, %215, %216, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %218 = stablehlo.dot_general %217, %arg203, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %219 = "mhlo.fusion"(%213, %arg204, %arg205, %214, %218, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %220 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %221 = "mhlo.fusion"(%arg208, %arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %222 = "mhlo.fusion"(%arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %223 = stablehlo.dot_general %219, %arg211, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %224 = "mhlo.fusion"(%arg212, %arg213, %222, %223, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %225 = "mhlo.fusion"(%arg212, %arg213, %222, %223, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %226 = stablehlo.dot_general %224, %225, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %227 = "mhlo.fusion"(%221, %226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %228 = "mhlo.fusion"(%227, %221, %226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %229 = "mhlo.fusion"(%228, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %230 = "mhlo.fusion"(%228, %229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %231 = "mhlo.fusion"(%arg212, %arg213, %222, %223, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %232 = stablehlo.dot_general %230, %231, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %234 = stablehlo.dot_general %233, %arg215, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %235 = "mhlo.fusion"(%219, %arg216, %arg217, %220, %234, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %236 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %237 = "mhlo.fusion"(%arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %238 = stablehlo.dot_general %235, %arg221, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %239 = "mhlo.fusion"(%arg222, %arg223, %237, %238, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %240 = stablehlo.dot_general %239, %arg225, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %241 = "mhlo.fusion"(%235, %arg226, %arg227, %236, %240, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %242 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %243 = "mhlo.fusion"(%arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %244 = "mhlo.fusion"(%arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %245 = stablehlo.dot_general %241, %arg233, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %246 = "mhlo.fusion"(%arg234, %arg235, %244, %245, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %247 = "mhlo.fusion"(%arg234, %arg235, %244, %245, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %248 = stablehlo.dot_general %246, %247, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %249 = "mhlo.fusion"(%243, %248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %250 = "mhlo.fusion"(%249, %243, %248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %251 = "mhlo.fusion"(%250, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %252 = "mhlo.fusion"(%250, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %253 = "mhlo.fusion"(%arg234, %arg235, %244, %245, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %254 = stablehlo.dot_general %252, %253, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %255 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %256 = stablehlo.dot_general %255, %arg237, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %257 = "mhlo.fusion"(%241, %arg238, %arg239, %242, %256, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %258 = "mhlo.fusion"(%arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %259 = "mhlo.fusion"(%arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %260 = stablehlo.dot_general %257, %arg243, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %261 = "mhlo.fusion"(%arg244, %arg245, %259, %260, %arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %262 = stablehlo.dot_general %261, %arg247, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %263 = "mhlo.fusion"(%257, %arg248, %arg249, %258, %262, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %264 = "mhlo.fusion"(%arg251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %265 = "mhlo.fusion"(%arg252, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16xf32>, %arg280: tensor<16x16xi32>):
      %c = stablehlo.constant dense<0> : tensor<i32>
      %289 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %290 = stablehlo.compare LT, %arg280, %289 : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %291 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<16x16xi32>
      %292 = stablehlo.add %arg280, %291 : tensor<16x16xi32>
      %293 = stablehlo.select %290, %292, %arg280 : tensor<16x16xi1>, tensor<16x16xi32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[256,1]{0,1}"} : (tensor<16x16xi32>) -> tensor<256x1xi32>
      %295 = mhlo.copy %294 : tensor<256x1xi32>
      %296 = "stablehlo.gather"(%arg279, %295) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x16xf32>, tensor<256x1xi32>) -> tensor<256x8x1xf32>
      mhlo.return %296 : tensor<256x8x1xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16xf32>, tensor<16x16xi32>) -> tensor<256x8x1xf32>
    %266 = "mhlo.fusion"(%arg254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<512xf32>
      %291 = stablehlo.rsqrt %290 : tensor<512xf32>
      mhlo.return %291 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %267 = stablehlo.dot_general %263, %arg255, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<512x384xf32>) -> tensor<16x512xf32>
    %268 = "mhlo.fusion"(%arg256, %arg257, %266, %267, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 0:16] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{3,1,2,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %269 = "mhlo.fusion"(%arg256, %arg257, %266, %267, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 16:32] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x16xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,16]{2,1,3,0}"} : (tensor<1x16x8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x16xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %301 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x16xf32>
    %270 = stablehlo.dot_general %268, %269, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x16xf32>) -> tensor<8x16x16xf32>
    %271 = "mhlo.fusion"(%265, %270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<256x8x1xf32>, %arg280: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg280, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg279 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_2) applies stablehlo.maximum across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %296 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16xf32>
    %272 = "mhlo.fusion"(%271, %265, %270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16xf32>, %arg280: tensor<256x8x1xf32>, %arg281: tensor<8x16x16xf32>):
      %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x16x16xf32>
      %290 = stablehlo.multiply %arg281, %289 : tensor<8x16x16xf32>
      %291 = mhlo.bitcast %arg280 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<256x8x1xf32>) -> tensor<16x16x8xf32>
      %292 = stablehlo.transpose %291, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,16,16]{0,2,1}"} : (tensor<16x16x8xf32>) -> tensor<8x16x16xf32>
      %293 = mhlo.copy %292 : tensor<8x16x16xf32>
      %294 = stablehlo.add %290, %293 : tensor<8x16x16xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
      %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x8x16xf32>
      %297 = stablehlo.maximum %296, %arg279 : tensor<1x8x16xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %300 = stablehlo.subtract %295, %299 : tensor<1x8x16x16xf32>
      %301 = stablehlo.exponential %300 : tensor<1x8x16x16xf32>
      mhlo.return %301 : tensor<1x8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16xf32>, tensor<256x8x1xf32>, tensor<8x16x16xf32>) -> tensor<1x8x16x16xf32>
    %273 = "mhlo.fusion"(%272, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<f32>):
      %289 = stablehlo.reduce(%arg279 init: %arg280) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
      mhlo.return %289 : tensor<1x8x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %274 = "mhlo.fusion"(%272, %273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1x8x16x16xf32>, %arg280: tensor<1x8x16xf32>):
      %289 = mhlo.bitcast %arg280 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16xf32>) -> tensor<8x16xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [1, 2] : (tensor<8x16xf32>) -> tensor<1x8x16x16xf32>
      %291 = stablehlo.divide %arg279, %290 : tensor<1x8x16x16xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x16xf32>) -> tensor<8x16x16xf32>
      mhlo.return %292 : tensor<8x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x8x16x16xf32>, tensor<1x8x16xf32>) -> tensor<8x16x16xf32>
    %275 = "mhlo.fusion"(%arg256, %arg257, %266, %267, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<512xf32>, %arg282: tensor<16x512xf32>, %arg283: tensor<512xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x512xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x512xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x512xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<512xf32>) -> tensor<16x512xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x512xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x512xf32>) -> tensor<1x16x8x64xf32>
      %298 = stablehlo.slice %297 [0:1, 0:16, 0:8, 32:64] : (tensor<1x16x8x64xf32>) -> tensor<1x16x8x32xf32>
      %299 = stablehlo.transpose %298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,8,16,32]{3,1,2,0}"} : (tensor<1x16x8x32xf32>) -> tensor<1x8x16x32xf32>
      %300 = mhlo.copy %299 : tensor<1x8x16x32xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x8x16x32xf32>) -> tensor<8x16x32xf32>
      mhlo.return %301 : tensor<8x16x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512xf32>, tensor<512xf32>) -> tensor<8x16x32xf32>
    %276 = stablehlo.dot_general %274, %275, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<8x16x32xf32>):
      %289 = mhlo.bitcast %arg279 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<1x8x16x32xf32>
      %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,8,32]{3,1,2,0}"} : (tensor<1x8x16x32xf32>) -> tensor<1x16x8x32xf32>
      %291 = mhlo.copy %290 : tensor<1x16x8x32xf32>
      %292 = mhlo.bitcast %291 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x16x8x32xf32>) -> tensor<1x16x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %295 = stablehlo.add %292, %294 : tensor<1x16x256xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %296 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %297 = stablehlo.clamp %293, %295, %296 : tensor<1x16x256xf32>
      %298 = stablehlo.multiply %292, %297 : tensor<1x16x256xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<1x16x256xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x256xf32>) -> tensor<16x256xf32>
      mhlo.return %301 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x16x32xf32>) -> tensor<16x256xf32>
    %278 = stablehlo.dot_general %277, %arg259, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x256xf32>, tensor<384x256xf32>) -> tensor<16x384xf32>
    %279 = "mhlo.fusion"(%263, %arg260, %arg261, %264, %278, %arg262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<16x384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg283, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg282, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg279, %296 : tensor<16x384xf32>
      mhlo.return %297 : tensor<16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<16x384xf32>
    %280 = "mhlo.fusion"(%arg263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<384xf32>
      %291 = stablehlo.rsqrt %290 : tensor<384xf32>
      mhlo.return %291 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %281 = "mhlo.fusion"(%arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<768xf32>
      %291 = stablehlo.rsqrt %290 : tensor<768xf32>
      mhlo.return %291 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %282 = stablehlo.dot_general %279, %arg265, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x384xf32>, tensor<768x384xf32>) -> tensor<16x768xf32>
    %283 = "mhlo.fusion"(%arg266, %arg267, %281, %282, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<768xf32>, %arg280: tensor<768xf32>, %arg281: tensor<768xf32>, %arg282: tensor<16x768xf32>, %arg283: tensor<768xf32>):
      %289 = stablehlo.broadcast_in_dim %arg283, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %290 = stablehlo.subtract %arg282, %289 : tensor<16x768xf32>
      %291 = stablehlo.broadcast_in_dim %arg281, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg280, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x768xf32>
      %295 = stablehlo.broadcast_in_dim %arg279, dims = [1] : (tensor<768xf32>) -> tensor<16x768xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x768xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16x768xf32>) -> tensor<1x16x768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %cst_2 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %300 = stablehlo.add %297, %299 : tensor<1x16x768xf32>
      %cst_3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %302 = stablehlo.clamp %298, %300, %301 : tensor<1x16x768xf32>
      %303 = stablehlo.multiply %297, %302 : tensor<1x16x768xf32>
      %cst_4 = stablehlo.constant dense<0.166666672> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x16x768xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x16x768xf32>
      %306 = mhlo.bitcast %305 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x768xf32>) -> tensor<16x768xf32>
      mhlo.return %306 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x768xf32>, tensor<768xf32>) -> tensor<16x768xf32>
    %284 = stablehlo.dot_general %283, %arg269, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<384x768xf32>) -> tensor<16x384xf32>
    %285 = "mhlo.fusion"(%arg270, %arg271, %arg272, %arg273, %279, %arg274, %arg275, %280, %284, %arg276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<384xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384xf32>, %arg282: tensor<384xf32>, %arg283: tensor<16x384xf32>, %arg284: tensor<384xf32>, %arg285: tensor<384xf32>, %arg286: tensor<384xf32>, %arg287: tensor<16x384xf32>, %arg288: tensor<384xf32>):
      %289 = stablehlo.broadcast_in_dim %arg288, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %290 = stablehlo.subtract %arg287, %289 : tensor<16x384xf32>
      %291 = stablehlo.broadcast_in_dim %arg286, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %292 = stablehlo.multiply %290, %291 : tensor<16x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg285, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<16x384xf32>
      %295 = stablehlo.broadcast_in_dim %arg284, dims = [1] : (tensor<384xf32>) -> tensor<16x384xf32>
      %296 = stablehlo.add %294, %295 : tensor<16x384xf32>
      %297 = stablehlo.add %arg283, %296 : tensor<16x384xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.reduce(%297 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<16x384xf32>, tensor<f32>) -> tensor<384xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-02> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<384xf32>
      %301 = stablehlo.subtract %300, %arg282 : tensor<384xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %303 = stablehlo.add %arg281, %302 : tensor<384xf32>
      %304 = stablehlo.rsqrt %303 : tensor<384xf32>
      %305 = stablehlo.multiply %301, %304 : tensor<384xf32>
      %306 = stablehlo.multiply %305, %arg280 : tensor<384xf32>
      %307 = stablehlo.add %306, %arg279 : tensor<384xf32>
      mhlo.return %307 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<16x384xf32>, tensor<384xf32>) -> tensor<384xf32>
    %286 = "mhlo.fusion"(%arg277) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg279: tensor<1000x384xf32>):
      %289 = stablehlo.transpose %arg279, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[384,1000]{0,1}"} : (tensor<1000x384xf32>) -> tensor<384x1000xf32>
      %290 = mhlo.copy %289 : tensor<384x1000xf32>
      mhlo.return %290 : tensor<384x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x384xf32>) -> tensor<384x1000xf32>
    %287 = "mhlo.fusion"(%arg278, %285, %286) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg279: tensor<1000xf32>, %arg280: tensor<384xf32>, %arg281: tensor<384x1000xf32>):
      %289 = stablehlo.dot %arg280, %arg281, precision = [DEFAULT, DEFAULT] : (tensor<384xf32>, tensor<384x1000xf32>) -> tensor<1000xf32>
      %290 = stablehlo.add %arg279, %289 : tensor<1000xf32>
      mhlo.return %290 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<384xf32>, tensor<384x1000xf32>) -> tensor<1000xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %288 : tensor<1x1000xf32>
  }
}
