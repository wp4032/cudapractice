#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "utils.cu"
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

template <typename T,
          size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K,
          size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y,
          size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y,
          size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y,
          size_t WMMA_TILE_SIZE_K,
          size_t NUM_THREADS>
__global__ void matmul_wmma_tiling(T const*a, T const*b, T *c, 
                                   size_t m, size_t n, size_t k, T alpha) {
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U, "BLOCK_TILE_SIZE_X must be divisible by WARP_TILE_SIZE_X");
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U, "BLOCK_TILE_SIZE_X must be divisible by WARP_TILE_SIZE_X");

    __shared__ T smem_a[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T smem_b[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];
    
    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U, "WARP_TILE_SIZE_X must be divisible by WMMA_TILE_SIZE_X");
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U, "WARP_TILE_SIZE_Y must be divisible by WMMA_TILE_SIZE_Y");
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U, "WARP_TILE_SIZE_K must be divisible by WMMA_TILE_SIZE_K");

    // Declaring fragments
    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T, wmma::col_major> a_frags[NUM_WMMA_TILES_Y];
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T, wmma::row_major> b_frags[NUM_WMMA_TILES_X];
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T> acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T> c_frag;

#pragma unroll
    for (size_t wmma_tile_row_idx = 0U; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx) {
        for (size_t wmma_tile_col_idx = 0U; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx) {
            wmma::fill_fragment(acc_frags, static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    size_t const warp_linear_idx = thread_linear_idx / 32U;
    size_t const warp_row_idx = warp_linear_idx / NUM_WARPS_X;
    size_t const warp_col_idx = warp_linear_idx % NUM_WARPS_X;

    size_t const num_thread_block_tiles = (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;

    for(size_t thread_block_tile_idx = 0U; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
        load_data_to_shared_memory_transposed_vectorized<
            T,
            BLOCK_TILE_SIZE_X,
            BLOCK_TILE_SIZE_Y,
            BLOCK_TILE_SIZE_K,
            NUM_THREADS,
            BLOCK_TILE_SKEW_SIZE_X,
            BLOCK_TILE_SKEW_SIZE_Y>
        (a, k, b, n, smem_a, smem_b, thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();
    }
    wmma::
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
    matmul_wmma_tiling<half, 128, 256, 32, 0, 0, 64, 32, 16, 32, 16, 8, 256><<<blocks, threads>>>(A, B, C, N, N, N, __float2half(1.0f));
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        // measure time
        matmul_wmma_tiling<half, 128, 256, 32, 0, 0, 64, 32, 16, 32, 16, 8, 256><<<blocks, threads>>>(A, B, C, N, N, N, __float2half(1.0f));
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
