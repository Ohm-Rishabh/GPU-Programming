#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  const int N = 1 << 16;
  size_t bytes = N * sizeof(int);
  int *a, *b, *c;

  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  int id = cudaGetDevice(&id);

  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, id);

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  
  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(a, bytes, id);
  cudaMemPrefetchAsync(b, bytes, id);
  
  int blk_size = 1 << 10;

  int grid_size = (N + blk_size - 1) / blk_size;

  vectorAdd<<<grid_size, blk_size>>>(a, b, c, N);

  cudaDeviceSynchronize();

  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
  return 0;
}
