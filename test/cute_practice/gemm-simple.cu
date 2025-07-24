#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <class TA, class TB, class TC>
void gemm_host(int M, int N, int K,
              TA const* A, int ldA,
              TB const* B, int ldB,
              TC      * C, int ldC) {
  float *nA = new float[M*K];
  float *nB = new float[N*K];
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      nA[i*K+k] = static_cast<float>(A[i*K+k]);
    }
  }
  for (size_t j = 0; j < N; j++) {
    for (size_t k = 0; k < K; k++) {
      nB[j*K+k] = static_cast<float>(B[j*K+k]);
    }
  }
  printf("finish host cast.\n");
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float ctemp = 0.0f;
      for (size_t k = 0; k < K; k++) {
        ctemp += nA[i*K+k] * nB[j*K+k];
      }
      C[i*N+j] = static_cast<TC>(ctemp);
    }
  }

  delete[] nA;
  delete[] nB;
}

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 1024;
  int n = 1024;
  int k = 1024;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  T *Cptr_host0;
  T *Cptr_host1;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  Cptr_host0 = (T*)malloc(sizeof(T) * m * n);
  Cptr_host1 = (T*)malloc(sizeof(T) * m * n);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  cudaMemcpy(Cptr_host0, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  gemm_host(m,n,k, Aptr_host, k, Bptr_host, k, Cptr_host1, n);
  float acc_host = 0;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      float v1 = Cptr_host0[i*n+j];
      float v2 = Cptr_host1[i*n+j];
      if (fabs(v1 - v2) > 0.2) {
        printf("v1 = %f, v2 = %f\n", v1, v2);
      }
      // acc_host += static_cast<float>(Cptr_host[i*n+j]);
      // printf("%f, ", static_cast<float>(h_C[i*n+j]));
    }
    // printf("\n");
  }
  // printf("mean: %f vs %f\n", acc_dev/(m*n), acc_host/(m*n));


  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // // cublas
  // T *Cptr_cublas;

  // cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  // cublasHandle_t handle;
  // cublasCreate(&handle);

  // half alpha = half(1.f);
  // half beta = half(0.f);
  // for (int i = 0; i < 100; ++i) {
  //   cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
  //         	  n, m, k,
  //         	  &alpha,
  //         	  (half *)Bptr, k,
  //         	  (half *)Aptr, k,
  //         	  &beta,
  //         	  (half *)Cptr_cublas, n);
  //   if (ret != CUBLAS_STATUS_SUCCESS) {
  //     printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
  //   }
  // }

  // cudaDeviceSynchronize();
  // err = cudaGetLastError();
  // printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // T *Cptr_host;
  // // T *Cptr_cublas_host;

  // Cptr_host = (T*)malloc(sizeof(T) * m * n);
  // // Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // // compare
  // cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  // cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  // float threshold = 0.1;
  // for (int i = 0; i < m * n; ++i) {
  //   float v1 = Cptr_host[i];
  //   // float v2 = Cptr_cublas_host[i];
  //   // if (fabs(v2 - v1) > threshold) {
  //   //   printf("v1 = %f, v2 = %f\n", v1, v2);
  //   // }
  // }

  // Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  // Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  // auto tile = make_tile(8, 8);
  // auto coor = make_coord(0, 0);
  // Tensor tc1 = local_tile(tensor_C, tile, coor);
  // Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  // print_tensor(tc1);
  // print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
