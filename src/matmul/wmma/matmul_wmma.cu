#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void init_matrix(half* A, half* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        A[IDX2C(row, col, N)] = __float2half(1.0f);
        B[IDX2C(row, col, N)] = __float2half(1.0f);
    }
}

__global__ void matmul_wmma(half *a, half *b, float *c, int N) {

  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  // Initialize fragment C to zero
  wmma::fill_fragment(c_frag, 0.0f);

  // Load inputs
  for (int tile_k = 0; tile_k < N / 16; ++tile_k) {
    // Calculate pointers to the current tile
    half* a_ptr = a + tile_row * 16 * N + tile_k * 16;
    half* b_ptr = b + tile_k * 16 * N + tile_col * 16;
    
    wmma::load_matrix_sync(a_frag, a_ptr, 16);
    wmma::load_matrix_sync(b_frag, b_ptr, 16);

    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* subC = c + tile_row * 16 * N + tile_col * 16;

  // Store output
  wmma::store_matrix_sync(subC, c_frag, N, wmma::mem_row_major);
}

int main() {
    int N = 1024;  // matrix dimension N×N
    size_t half_size = N * N * sizeof(half);
    size_t float_size = N * N * sizeof(float);

    half *A, *B;
    float *C;
    cudaMallocManaged(&A, half_size);
    cudaMallocManaged(&B, half_size);
    cudaMallocManaged(&C, float_size);

    dim3 threads(32);
    dim3 blocks(N / 16, N / 16);

    init_matrix<<<blocks, threads>>>(A, B, N);

    // initialize matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    // timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int repeats = 50;
    // warmup
    matmul_wmma<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        // measure time
        matmul_wmma<<<blocks, threads>>>(A, B, C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeats;

    // GFLOPS = (2 * N^3) / (time_in_seconds * 1e9)
    double tflops = (2.0 * N * N * N) / (ms / 1e3) / 1e12;
    printf("Matrix %d x %d\n", N, N);
    printf("Time: %.3f ms\n", ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // verify simple result
    printf("C[0] = %f\n", C[0]);

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
