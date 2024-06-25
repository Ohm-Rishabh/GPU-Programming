#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;
 
const int N = 1 << 10;
const int Mem_size = 1 << 10;

__global__ void matrixMul(const int *a, const int *b, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[Mem_size];
  __shared__ int s_b[Mem_size];

  int tmp = 0;
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  c[row * N + col] = tmp;
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  size_t bytes = N * N * sizeof(int);
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);


  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int threads = 32;
  int blks = N / THREADS;
  
  dim3 threads(threads, threads);
  dim3 blocks(blks, blks);
  
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c);
  return 0;
}
