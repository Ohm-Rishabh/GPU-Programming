#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::generate;
using std::vector;

__global__ void vectorAdd(int* a, int* b, int* c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;
  int *h_a, *h_b, *h_c;

  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);

  for(int i = 0; i < N; i++){
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }
  
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c, N);

  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
