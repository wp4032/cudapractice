#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// About cuBLAS
// cuBLAS is column major order 

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void init_matrix(float* A, float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        A[IDX2C(row, col, N)] = 1.0f;
        B[IDX2C(row, col, N)] = 1.0f;
    }
}

int main() {
    int N = 1024;  // matrix dimension N×N
    size_t size = N * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    init_matrix<<<blocks, threads>>>(A, B, N);

    // initialize matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int repeats = 50;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeats;

    // GFLOPS = (2 * N^3) / (time_in_seconds * 1e9)
    double gflops = (2.0 * N * N * N) / (ms / 1e3) / 1e9;
    printf("Matrix %d x %d\n", N, N);
    printf("Time: %.3f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // verify simple result
    printf("C[0] = %f\n", C[0]);

    cudaFree(A); cudaFree(B); cudaFree(C);
    cublasDestroy(handle);
    return 0;
}
