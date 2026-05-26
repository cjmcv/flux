

import numpy as np
import matplotlib.pyplot as plt


def dynamic_quant():
    N = 4096
    K = 4096
    
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
    
    # h20-gemmquant-15-4096
    torch_bf16    = [0.377, 0.78, 1.575, 3.13, 6.255, 12.34, 23.362, 39.492, 60.583, 82.915, 102.774, 117.797, 121.772, 128.134, 133.108]
    xop_bf16      = [0.559, 1.123, 2.251, 4.473, 9.107, 18.127, 36.341, 72.609, 93.666, 103.382, 125.062, 127.589, 135.107, 136.883, 137.246]
    xop_hopper_q4 = [1.002, 2.046, 4.081, 8.156, 15.621, 32.861, 60.088, 79.306, 98.184, 116.035, 123.832, 132.092, 132.001, 138.782, 139.046]
    xop_amphere_q4 = [1.427, 3.157, 6.196, 12.496, 24.786, 34.405, 46.321, 65.27, 76.96, 82.739, 85.982, 86.099, 86.07, 86.243, 86.245]
    xop_hopper_dynamic_q8 = [0.175, 0.351, 0.948, 1.888, 3.821, 7.627, 14.876, 29.378, 54.527, 85.179, 119.662, 156.494, 182.774, 200.567, 212.61]
    plt.plot(plot_x, torch_bf16, label='torch_bf16', marker='o', markersize=3)
    plt.plot(plot_x, xop_bf16, label='xop_bf16', marker='s', markersize=3)
    plt.plot(plot_x, xop_hopper_q4, label='xop_hopper_q4', marker='s', markersize=3)
    plt.plot(plot_x, xop_amphere_q4, label='xop_amphere_q4', marker='s', markersize=3)
    plt.plot(plot_x, xop_hopper_dynamic_q8, label='xop_hopper_dynamic_q8', marker='s', markersize=3)
    # plt.ylim(bottom=0)  # 

    plt.title(f'perf-N{N}-K{K}')
    plt.xlabel('m_size')
    plt.ylabel('tflops')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig('perf-N-{0}-K-{1}.png'.format(N, K))
    plt.show()
   
def rtx5090():
    N = 6144
    K = 1024
    
    exponent = 14 # 65536: 17
    for m in range(1, exponent):
        m = 2**m
        
    plot_x_value = [1] + list(2**x for x in list(range(1, exponent)))
    plot_x = range(len(plot_x_value))
    plt.xticks(plot_x, plot_x_value, rotation=45)

    tl_ws_bf16       = [1.543, 2.390, 4.858, 9.6790, 19.476, 39.171, 77.779, 98.308, 120.937, 157.334, 196.689, 209.567, 217.327, 221.361]
    tl_nows_bf16     = [1.543, 2.678, 5.271, 10.623, 21.249, 40.925, 84.455, 104.235, 119.12, 166.052, 204.469, 209.232, 217.785, 226.334]
    # torch_bf16    = [1.062, 2.035, 4.05, 8.202, 16.294, 32.207, 64.502, 96.191, 85.456, 121.034, 128.134, 168.44, 201.338, 205.272]
    torch_bf16    = [1.062, 2.035, 4.05, 8.202, 16.294, 32.207, 64.502, 96.191, 109.456, 121.034, 161.225, 168.44, 201.338, 205.272]
    xop_bf16      = [1.534, 2.716, 5.438, 11.01, 22.189, 39.925, 84.455, 112.238, 157.149, 196.597, 215.097, 225.433, 226.902, 231.862]
    # xop_bf16      = [1.534, 2.716, 5.438, 11.01, 22.189, 39.565, 78.646, 112.238, 157.149, 196.597, 215.097, 225.433, 226.902, 231.862]
    plt.plot(plot_x, xop_bf16, label='xop_bf16', marker='s', markersize=3)
    plt.plot(plot_x, tl_nows_bf16, label='xop_dsl_bf16', marker='s', markersize=3)
    plt.plot(plot_x, torch_bf16, label='torch_bf16', marker='o', markersize=3)
    
    # plt.plot(plot_x, xop_hopper_q4, label='xop_hopper_q4', marker='s', markersize=3)
    # plt.plot(plot_x, xop_amphere_q4, label='xop_amphere_q4', marker='s', markersize=3)
    # plt.plot(plot_x, xop_hopper_dynamic_q8, label='xop_hopper_dynamic_q8', marker='s', markersize=3)
    # plt.ylim(bottom=0)  # 

    plt.title(f'perf-N{N}-K{K}')
    plt.xlabel('m_size')
    plt.ylabel('tflops')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig('perf-N-{0}-K-{1}.png'.format(N, K))
    plt.show()
     
if __name__ == "__main__":
    # dynamic_quant()
    rtx5090()
