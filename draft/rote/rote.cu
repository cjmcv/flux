#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a+b-1)/(b))

//// vector add
// int block_size = 1024;
// int grid_size = CEIL(CEIL(N,4), 1024);
// elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
__global__ void elementwise_add_float4(float* a, float* b, float* c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N) return;
    
    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
}

// reduce sum
// constexpr size_t BLOCK_SIZE = 128;
// dim3 block_size(BLOCK_SIZE);
// dim3 grid_size(CEIL(CEIL(N, BLOCK_SIZE), 4));
__global__ void device_reduce_v5(float* d_x, float* d_y, const int N) {
	__shared__ float s_y[32];
	int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // 这里要乘以4
	int warpId = threadIdx.x / warpSize;   // 当前线程位于第几个warp
	int laneId = threadIdx.x % warpSize;   // 当前线程是warp中的第几个线程
	float val = 0.0f;
	if (idx < N) {
		float4 tmp_x = FLOAT4(d_x[idx]);
		val += tmp_x.x;
		val += tmp_x.y;
		val += tmp_x.z;
		val += tmp_x.w;
	}
	#pragma unroll
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	}

	if (laneId == 0) s_y[warpId] = val;
	__syncthreads();

	if (warpId == 0) {
		int warpNum = blockDim.x / warpSize;
		val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			val += __shfl_down_sync(0xFFFFFFFF, val, offset);
		}
		if (laneId == 0) atomicAdd(d_y, val);
	}
}
