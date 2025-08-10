#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
// #include <mpi.h>

// 编译执行方式参考:
// 编译,也可以使用g++,但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp9
// 运行 baseline
// ./outputfile baseline
// 运行 OpenMP
// ./outputfile openmp
// 运行子块并行优化
// ./outputfile block
// 运行MPI(假设4个进程)
// mpirun -np 4 ./outputfile mpi
// 运行MPI(假设4个进程)
// ./outputfile other
// 初始化矩阵(以一维数组形式表示),用于随机均填充浮点数


// 性能测量函数
double getMemoryUsage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0; // 转换为MB
    }
    return 0.0;
}


// 初始化矩阵(以一维数组形式表示),用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) 
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致,可以设计其他验证方法,来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6)
{
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法baseline实现(使用一维数组)
void matmul_baseline(const std::vector<double>& A,
                    const std::vector<double>& B,
                    std::vector<double>& C, 
                    int N, int M, int P) 
                    {
    for (int i = 0; i < N; ++i)
     {
        for (int j = 0; j < P; ++j) 
        {
            C[i*P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i*P + j] += A[i*M + k] * B[k*P + j];
        }
    }
}

// 方式1:利用OpenMP进行多线程并发的编程(主要修改函数)
void matmul_openmp(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P) 
                  {
    std::cout << "matmul openmp methods..." << std::endl;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < P; ++j) 
        {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) 
            {
                sum += A[i*M + k] * B[k*P + j];
            }
            C[i*P + j] = sum;
        }
    }

}


// 方式2:利用子块并行思想,进行缓存友好型的并行优化方法(主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        int N, int M, int P, 
                        int block_size = 64) 
                        {
    std::cout << "matmul block tiling methods..." <<std::endl;
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i += block_size) 
    {
        for (int j = 0; j < P; j += block_size)
         {
            for (int k = 0; k < M; k += block_size)
             {
                // 处理每个块
                for (int ii = i; ii < std::min(i + block_size, N); ++ii)
                 {
                    for (int jj = j; jj < std::min(j + block_size, P); ++jj) 
                    {
                        double sum = C[ii*P + jj];
                        for (int kk = k; kk < std::min(k + block_size, M); ++kk) 
                        {
                            sum += A[ii*M + kk] * B[kk*P + jj];
                        }
                        C[ii*P + jj] = sum;
                    }
                }
            }
        }
    }
}

// // 方式3:利用MPI消息传递,实现多进程并行优化 ((主要修改函数)
// void matmul_mpi(int N, int M, int P) {
//     // 待实现MPI通信和计算逻辑
//     std::cout << "[MPI] Method placeholder" << std::endl;
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // 计算每个进程处理的行数
//     int rows_per_proc = N / size;
//     int rows_remainder = N % size;
//     int my_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
//     int my_start_row = rank * rows_per_proc + std::min(rank, rows_remainder);

//     // 分配局部数组
//     std::vector<double> local_A(my_rows * M);
//     std::vector<double> B(M * P);
//     std::vector<double> local_C(my_rows * P, 0);

//     // 在根进程初始化数据
//     if (rank == 0) {
//         std::vector<double> A(N * M);
//         init_matrix(A, N, M);
//         init_matrix(B, M, P);

//         // 分发A的数据
//         int pos = 0;
//         for (int i = 0; i < size; ++i) {
//             int curr_rows = (i < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
//             if (i == 0) {
//                 std::copy(A.begin(), A.begin() + curr_rows * M, local_A.begin());
//             } else {
//                 MPI_Send(A.data() + pos * M, curr_rows * M, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
//             }
//             pos += curr_rows;
//         }
//     } else {
//         // 接收A的数据
//         MPI_Recv(local_A.data(), my_rows * M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     }

//     // 广播B到所有进程
//     MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     // 执行局部矩阵乘法
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < my_rows; ++i) {
//         for (int j = 0; j < P; ++j) {
//             double sum = 0.0;
//             for (int k = 0; k < M; ++k) {
//                 sum += local_A[i*M + k] * B[k*P + j];
//             }
//             local_C[i*P + j] = sum;
//         }
//     }

//     // 收集结果
//     if (rank == 0) {
//         std::vector<double> C(N * P);
//         std::copy(local_C.begin(), local_C.end(), C.begin());
        
//         // 接收其他进程的结果
//         int pos = my_rows;
//         for (int i = 1; i < size; ++i) {
//             int curr_rows = (i < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
//             MPI_Recv(C.data() + pos * P, curr_rows * P, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             pos += curr_rows;
//         }
//     } else {
//         // 发送结果给根进程
//         MPI_Send(local_C.data(), my_rows * P, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
//     }
// }


//方式4:其他方式(主要修改函数)
void matmul_other(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C, 
                int N, int M, int P)
                {
    std::cout << "Other methods..." << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
     {
        for (int j = 0; j < P; ++j)
         {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < M; ++k) 
            {
                sum += A[i*M + k] * B[k*P + j];
            }
            C[i*P + j] = sum;
        }
    }
}


int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    // // 处理MPI模式
    // if (mode == "mpi") {
    //     MPI_Init(&argc, &argv);
    //     matmul_mpi(N, M, P);
    //     MPI_Finalize();
    //     return 0;
    // }

    // 初始化矩阵数据
    std::vector<double> A(N*M);
    std::vector<double> B(M*P);
    std::vector<double> C(N*P, 0);
    std::vector<double> C_ref(N*P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);

    // 性能测量变量
    double start_time, end_time;
    double mem_start, mem_end;

    // 测量baseline性能
    mem_start = getMemoryUsage();
    start_time = omp_get_wtime();
    matmul_baseline(A, B, C_ref, N, M, P); // 生成基准结果
    end_time = omp_get_wtime();
    mem_end = getMemoryUsage();
    double baseline_time = end_time - start_time;
    double baseline_memory = mem_end - mem_start;

    // 执行指定模式
    if (mode == "baseline") 
    {
        std::cout << "[Baseline] Done.\n";
        std::cout << "[Baseline] Time: " << baseline_time << " seconds" << std::endl;
        std::cout << "[Baseline] Memory: " << baseline_memory << " MB" << std::endl;
    } 
    else if (mode == "openmp") 
    {
        mem_start = getMemoryUsage();
        start_time = omp_get_wtime();
        matmul_openmp(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        mem_end = getMemoryUsage();
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[OpenMP] Time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "[OpenMP] Speedup: " << baseline_time/(end_time - start_time) << "x" << std::endl;
        std::cout << "[OpenMP] Memory: " << mem_end - mem_start << " MB" << std::endl;
        std::cout << "[OpenMP] Memory rate: " << baseline_memory/(mem_end - mem_start) << "x" << std::endl;
    } 
    else if (mode == "block") 
    {
        mem_start = getMemoryUsage();
        start_time = omp_get_wtime();
        matmul_block_tiling(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        mem_end = getMemoryUsage();
        std::cout << "[Block_Parallel] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[Block_Parallel] Time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "[Block_Parallel] Speedup: " << baseline_time/(end_time - start_time) << "x" << std::endl;
        std::cout << "[Block_Parallel] Memory: " << mem_end - mem_start << " MB" << std::endl;
        std::cout << "[Block_Parallel] Memory rate: " << baseline_memory/(mem_end - mem_start) << "x" << std::endl;
    
    } 
    else if (mode == "other")
    {
        mem_start = getMemoryUsage();
        start_time = omp_get_wtime();
        matmul_other(A, B, C, N, M, P);
        end_time = omp_get_wtime();
        mem_end = getMemoryUsage();
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[Other] Time: " << end_time - start_time << " seconds" << std::endl;
        std::cout << "[Other] Speedup: " << baseline_time/(end_time - start_time) << "x" << std::endl;  
        std::cout << "[Other] Memory: " << mem_end - mem_start << " MB" << std::endl;
        std::cout << "[Other] Memory rate: " << baseline_memory/(mem_end - mem_start) << "x" << std::endl;
    }
    else 
    {
        std::cerr << "Usage: ./main [baseline|openmpblock|mpi)" << std::endl;
    }
    // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}