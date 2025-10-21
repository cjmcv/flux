

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    N = 2048
    K = 2048
    
    exponent = 15 # 65536: 17
    for m in range(1, exponent):
        m = 2**m
        
    plot_x_value = [1] + list(2**x for x in list(range(1, exponent)))
    plot_x = range(len(plot_x_value))
    plt.xticks(plot_x, plot_x_value, rotation=45)
    
    # xop_perf   = [0.5, 1.0, 2.0, 5.0, 10,  18,  30, 50,  78, 103, 111, 107, 103, 102, 103 ]    # gemmcomm_8192_sm89-l40x2
    # torch_perf = [0.5, 1.0, 2.0, 4.9, 9, 16.5,  28, 48,  75,  82, 82,   83, 85.5, 85.5, 65 ]
    # xop_perf   = [0.6,  1.1,  2.5,  6.0,  10,   19,  31,  46.5, 51.5,  56,  65,   67,   65,   61,   62 ]      # gemmcomm_4096_sm89-l40x2
    # torch_perf = [0.5,  1.0,  2.0,  5.0,  8,   12,   20,   32,   48,    54,  56,   55,   58,  58,   59 ]    
    
    # xop_perf = [0.3871200339852235, 0.7782001014947636, 1.5609315220333064, 3.102079919665585, 6.169314558041775, 12.28645376448096, 24.31421043255768, 47.50632518323308, 90.54736622006526, 97.64563186927019, 102.04290028737228, 117.07877252948464, 118.63773457572, 123.48474592936866, 125.86685421779254]
    # torch_perf = [0.3791930172580574, 0.7554347201037467, 1.509653937842081, 2.9996001247971864, 5.967403772850797, 11.791071154262328, 23.122062006779597, 44.43610880483963, 76.95994236668582, 89.66848436775503, 97.51349081524857, 113.85529661737344, 116.46612904142223, 122.34146571519419, 125.34037497118736]

    
    # #              1     2    4   8   16   32   64   128   256  512  1024  2048  4096  8192  16384
    # xop_perf   = [0.6,  0.8,  2,  3,  5,   8,   13,  22,  27,  31,   33,   34.7,   35,  34.7,   34.7 ]      # gemmcomm_2048_sm89-l40x2
    # torch_perf = [0.5,  0.7,  1,  2,  3,   6,   11,  16,   24,  30,  32,   33.8,   34,  34.7,   34.7 ]    
    plt.plot(plot_x, xop_perf, label='xop_fused_gemmcomm', marker='o', markersize=3)
    plt.plot(plot_x, torch_perf, label='xop_gemm+nccl', marker='s', markersize=3)
    
    # plt.ylim(bottom=0)  # 

    plt.title(f'perf-N{N}-K{K}')
    plt.xlabel('m_size')
    plt.ylabel('tflops')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig('perf-N-{0}-K-{1}.png'.format(N, K))
    plt.show()
