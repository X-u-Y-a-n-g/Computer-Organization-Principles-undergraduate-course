#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward

#define BATCH 1024
#define I 10
#define H 20
#define O 5
#define BLOCK_SIZE 16 // 添加块大小定义

__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void add_bias_kernel(double* C, const double* bias, int M, int N) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;
    
    if (row < M && col < N) {
        C[row * N + col] += bias[col];
    }
}

__global__ void relu_kernel(double* A, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

void random_init(std::vector<double>& mat) 
{
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
}

int main() {
    // 初始化主机端数据
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O);

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 声明设备端指针
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;

    // 分配设备内存
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    // 定义线程块和网格维度
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    // 第一个矩阵乘法的网格维度 (BATCH x H)
    dim3 gridDim1(
        (H + BLOCK_SIZE - 1) / BLOCK_SIZE,      
        (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE   
    );
    
    // 第二个矩阵乘法的网格维度 (BATCH x O)
    dim3 gridDim2(
        (O + BLOCK_SIZE - 1) / BLOCK_SIZE,      
        (BATCH + BLOCK_SIZE - 1) / BLOCK_SIZE   
    );

    // 添加计时器
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float milliseconds = 0;

    // 记录总体开始时间
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 数据传输到设备
    hipEventRecord(start);
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Data transfer H2D time: " << milliseconds << " ms" << std::endl;

    // 记录计算时间
    hipEventRecord(start);
    
    // Hidden layer: H = X * W1
    matmul_kernel<<<gridDim1, blockDim>>>(d_X, d_W1, d_H, BATCH, H, I);
    add_bias_kernel<<<(BATCH + 255) / 256, dim3(256, H)>>>(d_H, d_B1, BATCH, H);
    relu_kernel<<<(BATCH * H + 255) / 256, 256>>>(d_H, BATCH * H);
    
    // Output layer: Y = H * W2
    matmul_kernel<<<gridDim2, blockDim>>>(d_H, d_W2, d_Y, BATCH, O, H);
    add_bias_kernel<<<(BATCH + 255) / 256, dim3(256, O)>>>(d_Y, d_B2, BATCH, O);

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Computation time: " << milliseconds << " ms" << std::endl;

    // 数据传回主机
    hipEventRecord(start);
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Data transfer D2H time: " << milliseconds << " ms" << std::endl;

    // 计算总时间和性能指标
    auto total_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // 计算总操作次数和GFLOPS
    double total_ops = BATCH * (
        2.0 * I * H +      // 第一个矩阵乘法
        H +                // 第一个偏置加法
        H +                // ReLU操作
        2.0 * H * O +      // 第二个矩阵乘法
        O                  // 第二个偏置加法
    );
    
    double seconds = duration.count() / 1e6;
    double gflops = (total_ops / seconds) / 1e9;
    
    // 输出性能统计
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "Total time: " << seconds * 1000 << " ms" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;

    // 打印部分输出结果
    std::cout << "\nOutput samples:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]:";
        for (int j = 0; j < O; ++j) {
            std::cout << " " << h_Y[i * O + j];
        }
        std::cout << std::endl;
    }
    
    // 清理资源
    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    return 0;
}