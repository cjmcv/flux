
import os
import sys
import argparse
import itertools

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(CURRENT_PATH + "/../")
# print(sys.path)

class SearchSpaceGenerator:
    # ctlop type warp
    def xtw(self, type):
        return "(int8_t)ME::" + type
    # cutlass type warp
    def ctw(self, type):
        return "cutlass::" + type
    # cutlass shape type warp
    def cstw(self, shape):
        return 'cutlass::gemm::GemmShape<{0},{1},{2}>'.format(str(shape[0]), str(shape[1]), str(shape[2]))

    def ctlop_to_cutlasstype(self, ctlop_type):
        string_to_string = {
            "Normal": "Error",
            "Void": "Void",
            "FP16": "cutlass::half_t",
            "BF16": "cutlass::bfloat16_t",
            "FP32": "float",
            "E4M3": "cutlass::float_e4m3_t",
            "E5M2": "cutlass::float_e5m2_t",
            "S8": "int8_t",
            "S32": "int32_t",
            "Sm80": "cutlass::arch::Sm80",
            "Sm89": "cutlass::arch::Sm89",
            "Sm90": "cutlass::arch::Sm90",
            "RRR": "cutlass::layout::RowMajor, layout::RowMajor, layout::RowMajor",
            "RCR": "cutlass::layout::RowMajor, layout::ColumnMajor, layout::RowMajor",
            "RCC": "cutlass::layout::RowMajor, layout::ColumnMajor, layout::ColumnMajor",
            #
            "Identity": "gemm::threadblock::GemmIdentityThreadblockSwizzle<>",
            "StreamK": "gemm::threadblock::ThreadblockSwizzleStreamK",
        }
        # 返回对应的字符串，如果没有匹配的值，则返回"Unknown"
        return string_to_string.get(ctlop_type, "Unknown")
    
    def get_meta_space(self):
        data_type = [('FP16', 'FP16', 'FP16', 'FP32')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm80'] # , 'Sm89'

        res = []
        for t, l, a in itertools.product(data_type, layout, arch):
            meta_ctlop_str = ''
            meta_cutlass_str = ''
            for ti in t:
                meta_ctlop_str += self.xtw(ti) + ', '
                meta_cutlass_str += self.ctlop_to_cutlasstype(ti) + ', '

            meta_ctlop_str += self.xtw(l) + ', '
            meta_cutlass_str += self.ctlop_to_cutlasstype(l) + ', '

            meta_ctlop_str += self.xtw(a)
            meta_cutlass_str += self.ctlop_to_cutlasstype(a)

            res.append((meta_ctlop_str, meta_cutlass_str))
        return res

    def get_hparam_space(self):
        # block_shapes = [(128, 128, 32), (64, 256, 32)]
        # warp_shapes = [(64, 64, 32)]
        # instruction_shapes = [(16, 8, 16), (16, 8, 8)]
        bwi_shapes = [((128, 128, 32), (64, 64, 32), (16, 8, 16)),
                      ((64, 256, 32), (64, 64, 32), (16, 8, 16)),
                      ((64, 128, 32), (32, 64, 32), (16, 8, 16)),
                      ((64, 64, 32), (32, 32, 32), (16, 8, 8))]

        swizzles = ['Identity', 'StreamK']
        stages = [3, 4]
        splitk_factors = [1, 2]
        avail_smss = [-1, 1]

        res = []
        for bwi_shape, swizzle, stage, splitk_factor, avail_sm in itertools.product(
            bwi_shapes, swizzles, stages, splitk_factors, avail_smss):
            bshape = bwi_shape[0]
            wshape = bwi_shape[1]
            ishape = bwi_shape[2]
            # Ignore special case.
            if (swizzle == 'Identity' and avail_sm != 1):
                continue
            hparam_str = '{0},{1},{2},{3},{4},{5},{6}'.format(
                self.cstw(bshape), self.cstw(wshape), self.cstw(ishape), self.ctlop_to_cutlasstype(swizzle), str(stage), str(splitk_factor), str(avail_sm))
            
            res.append(hparam_str)
        return res

    def run(self, tag):
        fp = {}
        fp[tag] = open("config_{0}_sm89.cu".format(tag.lower()), "w")
        fp[tag].write('// clang-format off\n')
        fp[tag].write('#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"\n')
        fp[tag].write('#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"\n\n')
        fp[tag].write('namespace ctlop {\n')
        fp[tag].write('using namespace cutlass;\n')
        fp[tag].write('using ME = UnifiedMetaEnum;\n\n')
        fp[tag].write('static int config_{0}_sm89 = []() {{\n'.format(tag.lower()))
        fp[tag].write('  GemmConfigRegister& ins = GemmConfigRegister::instance();\n')

        ctlop_tag = self.xtw(tag)
        meta = self.get_meta_space()
        hparam = self.get_hparam_space()
        for m in meta:
            ctlop_meta, cutlass_meta = m
            for id, h in enumerate(hparam):
                fp[tag].write('  ins.add({{{0},{1},{2}}}, '.format(str(id), ctlop_tag, ctlop_meta))
                fp[tag].write('/*op*/[]() {{ return new GemmPureV2Impl</*meta*/{0},/*hparam*/{1}>();}});\n'.format(cutlass_meta, h))

        fp[tag].write('  return 0;\n}();\n}')
        fp[tag].write('// clang-format on')
        
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--model_path", type=str, default="../models/tf_micro_conv_test_model.int8.tflite")
    parser.add_argument("--output_path", type=str, default="./")
    args = parser.parse_args()

    (filepath, filename) = os.path.split(args.model_path)
    model_tag = os.path.splitext(filename)[0]+"_model"
    model_tag = model_tag.replace('.', '_')

    generator = SearchSpaceGenerator()
    generator.run("Normal") 