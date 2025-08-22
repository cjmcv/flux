
import os
import sys
import argparse
import itertools

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(CURRENT_PATH + "/../")
# print(sys.path)

class TypeWarpper:
    def tag_unify(self, tag):
        if (tag == "GemmV2BlockScaleFp8"):
            return "GemmBlockScaleFp8"
        return tag
    # xop type warp
    def xtw(self, type):
        return "(int16_t)ME::" + type
    # cutlass type warp
    def ctw(self, type):
        return "cutlass::" + type
    # cutlass shape type warp
    def cstw(self, shape, version=2):
        if version == 2:
            return 'cutlass::gemm::GemmShape<{0},{1},{2}>'.format(str(shape[0]), str(shape[1]), str(shape[2]))
        else:
            return 'cute::Shape<cute::_{0},cute::_{1},cute::_{2}>'.format(str(shape[0]), str(shape[1]), str(shape[2]))

    def xop_to_cutlasstype(self, xop_type):
        string_to_string = {
            "GemmNormal": "Error",
            "GemmNormalSimt": "Error",
            "Void": "Void",
            "FP16": "cutlass::half_t",
            "BF16": "cutlass::bfloat16_t",
            "FP32": "float",
            "E4M3": "cutlass::float_e4m3_t",
            "E5M2": "cutlass::float_e5m2_t",
            "S8": "int8_t",
            "S32": "int32_t",
            "Sm80": "arch::Sm80",
            "Sm89": "arch::Sm89",
            "Sm90": "arch::Sm90",
            #
            "RRR": "layout::RowMajor, layout::RowMajor, layout::RowMajor",
            "RCR": "layout::RowMajor, layout::ColumnMajor, layout::RowMajor",
            "RCC": "layout::RowMajor, layout::ColumnMajor, layout::ColumnMajor",
            #
            "SwizzleIdentity": "gemm::threadblock::GemmIdentityThreadblockSwizzle<>",
            "SwizzleStreamK": "gemm::threadblock::ThreadblockSwizzleStreamK",
            #
            "Heuristic": "xop::RasterOrderOptions::Heuristic",
            "AlongM": "xop::RasterOrderOptions::AlongM",
            "AlongN": "xop::RasterOrderOptions::AlongN",
            # 
            "TSPersistent": "cutlass::gemm::PersistentScheduler",
            "TSStreamK": "cutlass::gemm::StreamKScheduler"
        }
        # Return the corresponding string, or return "Unknown" if there is no matching value.
        return string_to_string.get(xop_type, "Unknown")

def make_meta_space(w, data_type, layout, arch):
    res = []
    for t, l, a in itertools.product(data_type, layout, arch):
        meta_xop_str = ''
        meta_cutlass_str = ''
        for ti in t:
            meta_xop_str += w.xtw(ti) + ', '
            meta_cutlass_str += w.xop_to_cutlasstype(ti) + ', '

        meta_xop_str += w.xtw(l) + ', '
        meta_cutlass_str += w.xop_to_cutlasstype(l) + ', '

        meta_xop_str += w.xtw(a)
        meta_cutlass_str += w.xop_to_cutlasstype(a)

        res.append((meta_xop_str, meta_cutlass_str))
    return res

class GemmNormalSchema:
    impl = "GemmPureV2Impl"
    impl_header = "gemm_normal/gemm_v2_impl.h"
    
    def get_meta_space(self, w):
        # ('BF16', 'BF16', 'BF16', 'FP32'), ('FP16', 'FP16', 'FP16', 'FP32'), ('FP16', 'FP16', 'FP16', 'FP16')
        data_type = [('BF16', 'BF16', 'BF16', 'FP32')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm80'] # , 'Sm89'

        res = make_meta_space(w, data_type, layout, arch)
        return res

    def get_hparam_space(self, w):
        # block_shapes = [(128, 128, 32), (64, 256, 32)]
        # warp_shapes = [(64, 64, 32)]
        # instruction_shapes = [(16, 8, 16), (16, 8, 8)] 
        # Note: These shapes may fail the can_implement in some cases, 
        #     which is presumably related to resource limitations, 
        #     as the block tile exceeds four times the warp tile.
        # ((128, 256, 32), (64, 64, 32), (16, 8, 16)), 
        # ((128, 128, 64), (64, 64, 32), (16, 8, 16)),
        # ((64, 256, 64), (64, 64, 32), (16, 8, 16)),
        bwi_shapes = [((128, 128, 32), (64, 64, 32), (16, 8, 16)),
                      ((64, 256, 32), (64, 64, 32), (16, 8, 16)),
                      ((64, 128, 64), (64, 64, 32), (16, 8, 16)),
                      ((64, 128, 32), (64, 64, 32), (16, 8, 16)),
                      ((64, 128, 32), (32, 64, 32), (16, 8, 16)),
                      ((32, 128, 64), (16, 64, 64), (16, 8, 16)),
                      ((32, 128, 64), (16, 64, 32), (16, 8, 16)),
                      ((16, 128, 64), (16, 64, 64), (16, 8, 16)),
                      ((16, 128, 64), (16, 64, 32), (16, 8, 16)),
                      ((16, 128, 64), (16, 64, 32), (16, 8, 8))]
        swizzles = ['SwizzleIdentity', 'SwizzleStreamK']
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
            if (swizzle == 'SwizzleIdentity' and avail_sm == 1):
                continue
            if (swizzle == 'SwizzleIdentity' and stage == 4 and splitk_factor == 2):
                continue
            hparam_str = '{0},{1},{2},{3},{4},{5},{6}'.format(
                w.cstw(bshape), w.cstw(wshape), w.cstw(ishape), w.xop_to_cutlasstype(swizzle), str(stage), str(splitk_factor), str(avail_sm))
            
            res.append(hparam_str)
        return res
class GemmNormalSimtSchema:
    impl = "GemmPureV2SimtImpl"
    impl_header = "gemm_normal/gemm_v2_simt_impl.h"
    
    def get_meta_space(self, w):
        data_type = [('BF16', 'BF16', 'BF16', 'FP32')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm80'] # , 'Sm89'

        res = make_meta_space(w, data_type, layout, arch)
        return res

    def get_hparam_space(self, w):
        # block_shapes = [(128, 128, 32), (64, 256, 32)]
        # warp_shapes = [(64, 64, 32)] 
        bw_shapes = [((64, 64, 4), (32, 16, 4)),
                     ((32, 32, 4), (16, 16, 4)),
                     ((16, 32, 4), (8, 16, 4))]
        swizzles = ['SwizzleIdentity', 'SwizzleStreamK']
        stages = [3, 4]
        splitk_factors = [1, 2]

        res = []
        for bw_shape, swizzle, stage, splitk_factor in itertools.product(
            bw_shapes, swizzles, stages, splitk_factors):
            bshape = bw_shape[0]
            wshape = bw_shape[1]
            hparam_str = '{0},{1},{2},{3},{4}'.format(
                w.cstw(bshape), w.cstw(wshape), w.xop_to_cutlasstype(swizzle), str(stage), str(splitk_factor))
            
            res.append(hparam_str)
        return res
    
class GemmV2BlockScaleFp8Schema:
    impl = "GemmV2BlockScaleFp8Impl"
    impl_header = "gemm_normal/gemm_v2_blockscale_fp8_impl.h"
    
    def get_meta_space(self, w):
        # ('E4M3', 'E4M3', 'FP16', 'FP16')
        # data_type = [('E4M3', 'E4M3', 'BF16', 'FP32')] # a,b,cd,acc
        data_type = [('E4M3', 'E4M3', 'BF16', 'FP32'), ('E4M3', 'E4M3', 'BF16', 'FP16'),
                     ('E4M3', 'E4M3', 'FP16', 'FP32'), ('E4M3', 'E4M3', 'FP16', 'FP16')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm89']
        res = make_meta_space(w, data_type, layout, arch)
        return res

    def get_hparam_space(self, w):
        tile_shapes = [(32, 128, 128), (64, 128, 128), (128, 128, 128), (16, 128, 128)]
        perm_shapes = [(32, 32, 32), (32, 64, 64), (16, 32, 32)]
        stages = [3, 4]

        res = []
        for tile_shape, perm_shape, stage in itertools.product(
            tile_shapes, perm_shapes, stages):
            if (perm_shape[0] > tile_shape[0]):
                continue
            if (perm_shape[0]==16 and tile_shape[0]!=16):
                continue
            if (tile_shape[0]==128 and stage==4):
                continue
            hparam_str = '{0},{1},{2}'.format(
                w.cstw(tile_shape,3), w.cstw(perm_shape,3), str(stage))
            res.append(hparam_str)
        return res
    
class GemmBlockScaleFp8Schema:
    impl = "GemmBlockScaleFp8Impl"
    impl_header = "gemm_normal/gemm_v3_blockscale_fp8_impl.h"
    
    def get_meta_space(self, w):
        data_type = [('E4M3', 'E4M3', 'BF16', 'FP32')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm90']
        res = make_meta_space(w, data_type, layout, arch)
        return res

    def get_hparam_space(self, w):
        tile_schedulers = ["TSPersistent", "TSStreamK"]
        tile_shapes = [(128, 128, 128)]
        cluster_shapes = [(1, 2, 1), (2, 1, 1)]
        raster_orders = ["Heuristic", "AlongM", "AlongN"]
        swizzles = [2,4,8] # 1,2,4,8

        res = []
        for tile_scheduler, tile_shape, cluster_shape, raster_order, swizzle in itertools.product(
            tile_schedulers, tile_shapes, cluster_shapes, raster_orders, swizzles):
            hparam_str = '{0},{1},{2},{3},{4}'.format(
                w.xop_to_cutlasstype(tile_scheduler), w.cstw(tile_shape,3), w.cstw(cluster_shape,3),
                w.xop_to_cutlasstype(raster_order), str(swizzle))
            res.append(hparam_str)
        return res
    
class GemmGroupedBolckScaleFp8Schema:
    impl = "GemmGroupedBlockScaleFp8Impl"
    impl_header = "gemm_normal/gemm_v3_grouped_blockscale_fp8_impl.h"
    
    def get_meta_space(self, w):
        data_type = [('E4M3', 'E4M3', 'BF16', 'FP32')] # a,b,cd,acc
        layout = ['RCR'] # , 'RRR'
        arch = ['Sm90']
        res = make_meta_space(w, data_type, layout, arch)
        return res

    def get_hparam_space(self, w):
        tile_schedulers = ["TSPersistent", "TSStreamK"]
        tile_shapes = [(128, 128, 128)]
        cluster_shapes = [(1, 2, 1), (2, 1, 1)]
        raster_orders = ["Heuristic", "AlongM", "AlongN"]
        swizzles = [2,4,8] # 1,2,4,8

        res = []
        for tile_scheduler, tile_shape, cluster_shape, raster_order, swizzle in itertools.product(
            tile_schedulers, tile_shapes, cluster_shapes, raster_orders, swizzles):
            hparam_str = '{0},{1},{2},{3},{4}'.format(
                w.xop_to_cutlasstype(tile_scheduler), w.cstw(tile_shape,3), w.cstw(cluster_shape,3),
                w.xop_to_cutlasstype(raster_order), str(swizzle))
            res.append(hparam_str)
        return res
        
def str2schema(schema_name):
    string_to_schema = {
        "GemmNormal": GemmNormalSchema(),
        "GemmNormalSimt": GemmNormalSimtSchema(),
        "GemmV2BlockScaleFp8": GemmV2BlockScaleFp8Schema(),
        "GemmBlockScaleFp8": GemmBlockScaleFp8Schema(),
        "GemmGroupedBlockScaleFp8": GemmGroupedBolckScaleFp8Schema(),
    }
    return string_to_schema.get(schema_name, None)

class SearchSpaceGenerator:
    def run(self, tag, output_path):
        schema = str2schema(tag)

        fp = {}
        fp[tag] = open(output_path + "/search_space_{0}.cu".format(tag.lower()), "w")
        fp[tag].write('// clang-format off\n')
        fp[tag].write('#include "xop/ops_impl/{0}"\n\n'.format(schema.impl_header))
        fp[tag].write('namespace xop {\n')
        fp[tag].write('using namespace cutlass;\n')
        fp[tag].write('using ME = UnifiedMetaEnum;\n\n')
        fp[tag].write('static int search_space_{0} = []() {{\n'.format(tag.lower()))
        fp[tag].write('  GemmConfigRegister& ins = GemmConfigRegister::instance();\n')

        type_warpper = TypeWarpper()
        xop_tag = type_warpper.xtw(type_warpper.tag_unify(tag))
        meta = schema.get_meta_space(type_warpper)
        hparam = schema.get_hparam_space(type_warpper)
        for m in meta:
            xop_meta, cutlass_meta = m
            for id, h in enumerate(hparam):
                fp[tag].write('  ins.add({{{0},{1},{2}}}, '.format(str(id), xop_tag, xop_meta))
                fp[tag].write('/*op*/[]() {{ return new {0}</*meta*/{1},/*hparam*/{2}>();}});\n'.format(schema.impl, cutlass_meta, h))

        fp[tag].write('  return 0;\n}();\n}')
        fp[tag].write('// clang-format on')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--schema", type=str, default="None")
    parser.add_argument("--output_path", type=str, default="./tools/") # ./src/ops/gemm_normal/tuning_config/
    args = parser.parse_args()

    if (args.schema == "None"):
        print("usage: python3 tools/gemm/gen_search_space.py --schema=GemmNormal (GemmNormal/GemmNormalSimt/GemmV2BlockScaleFp8/GemmBlockScaleFp8/GemmGroupedBlockScaleFp8)")
        exit()
    generator = SearchSpaceGenerator()
    generator.run(args.schema, args.output_path) 