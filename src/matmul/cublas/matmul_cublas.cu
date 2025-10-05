#include <stdio.h>
#include <cuda_runtime.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

__global__ void matmul_basic(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

int main() {
    int N = 1024;  // matrix dimension N×N
    size_t size = N * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // initialize matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_basic<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // GFLOPS = (2 * N^3) / (time_in_seconds * 1e9)
    double gflops = (2.0 * N * N * N) / (ms / 1e3) / 1e9;
    printf("Matrix %d x %d\n", N, N);
    printf("Time: %.3f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // verify simple result
    printf("C[0] = %f\n", C[0]);

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
