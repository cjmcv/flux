import unittest

import numpy as np
import torch
import torch.nn as nn

import xop

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')

class Test(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        w = torch.randn((n, k), dtype=torch.half, device=DEV)
        B_ref = w.t()
        _, B, s = xop.marlin_quant_int4(w, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        xop.marlin_fp16xint4_matmul(A, B, C, s, workspace, thread_k, thread_n, -1, 16)
        torch.cuda.synchronize()
        print(C)
        print(C_ref)
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 1)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        return
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(128, 128)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0], thread_k, thread_n)

    def test_errors(self):
        print()
        m, n, k = 16, 256, 64
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        w = torch.randn((k, n), dtype=torch.half, device=DEV)
        B_ref, B, s = xop.marlin_quant_int4(w)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            xop.marlin_fp16xint4_matmul(A, B, C, s, workspace, 128, 128, -1, 16)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            xop.marlin_fp16xint4_matmul(A, B, C, s, workspace, 256, 256, -1, 16)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            xop.marlin_fp16xint4_matmul(A, B, C, s, workspace, 256, 256, -1, 16)
        except:
            err = True 
        self.assertTrue(err)

    def test_groups(self):
        print()
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)


if __name__ == '__main__':
    unittest.main()
