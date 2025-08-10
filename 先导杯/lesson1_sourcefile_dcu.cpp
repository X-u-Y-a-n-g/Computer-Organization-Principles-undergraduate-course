#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// 编译命令
// hipcc source_file_dau.cpp -o outputfile_dcu
// 执行命令
// ./outputfile_dou

#define N 1024
#define M 2024
#define P 512

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }


    return;
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
    return;
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
    return;
}

bool validate(const std::vector<double>& ref,  const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    }
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_nodef(N * P);
    init_matrix(A);
    init_matrix(B);

    // CPU baseline
    matmul_cpu(A, B, C);

    //主要修改部分
    // Allocate and copy to device, use matmul kernel to compute in DCU
    // 设备内存分配
    double *d_A, *d_B, *d_C;
    hipMalloc((void**)&d_A, N * M * sizeof(double));
    hipMalloc((void**)&d_B, M * P * sizeof(double));
    hipMalloc((void**)&d_C, N * P * sizeof(double));
    // 将数据从主机复制到设备
    hipMemcpy(d_A, A.data(), N * M * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), M * P * sizeof(double), hipMemcpyHostToDevice);

    // 配置kernel启动参数
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((P + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动kernel
    hipLaunchKernelGGL(matmul_kernel, 
                       gridDim, blockDim,
                       0, 0,
                       d_A, d_B, d_C, N, M, P);

    // 同步设备
    hipDeviceSynchronize();

    // 将结果从设备复制回主机
    std::vector<double> C_gpu(N * P);
    hipMemcpy(C_gpu.data(), d_C, N * P * sizeof(double), hipMemcpyDeviceToHost);

    // 验证结果
    if (validate(C, C_gpu)) {
        std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
        std::cout << "[HIP] Valid: 0" << std::endl;
    }

    // 添加性能评测代码
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // 测量GPU执行时间
    hipEventRecord(start);
    hipLaunchKernelGGL(matmul_kernel, 
                       gridDim, blockDim,
                       0, 0,
                       d_A, d_B, d_C, N, M, P);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Execution Time: " << milliseconds << " ms" << std::endl;

    // 清理资源
    hipEventDestroy(start);
    hipEventDestroy(stop);

    // 释放设备内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // 需额外增加性能评测代码
    return 0;
}