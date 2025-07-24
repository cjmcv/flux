#include <iostream>

// 定义一个简单的 CUDA 核函数
__global__ void addKernel(int *c, const int *a, const int *b) {
    int index = threadIdx.x; // 获取线程索引
    c[index] = a[index] + b[index]; // 执行加法操作
}

int main() {
    int a[256], b[256], c[256]; // 定义数组
    int *dev_a, *dev_b, *dev_c; // 定义设备指针

    // 初始化输入数组
    for (int i = 0; i < 256; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 分配设备内存
    cudaMalloc((void**)&dev_a, 256 * sizeof(int));
    cudaMalloc((void**)&dev_b, 256 * sizeof(int));
    cudaMalloc((void**)&dev_c, 256 * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(dev_a, a, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数
    addKernel<<<1, 256>>>(dev_c, dev_a, dev_b);

    // 将结果从设备复制回主机
    cudaMemcpy(c, dev_c, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // 打印结果
    for (int i = 0; i < 256; ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    return 0;
}