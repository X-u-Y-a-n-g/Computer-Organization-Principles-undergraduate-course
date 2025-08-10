#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>

// 预定义参数
#define INPUT_DIM 10
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 200
#define LEARNING_RATE 1e-4

// 矩阵乘法 kernel
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ReLU前向传播
__global__ void relu_forward(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]);
    }
}

// MSE损失计算
__global__ void compute_mse_loss(const double* pred, const double* target, 
                               double* loss, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        atomicAdd(loss, diff * diff / (2.0 * size));
        grad[idx] = diff / size;
    }
}

// ReLU反向传播
__global__ void compute_relu_backward(double* grad, const double* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (input[idx] > 0 ? 1.0 : 0.0);
    }
}

// 权重更新
__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// 矩阵转置
__global__ void transpose(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            token.erase(std::remove_if(token.begin(), token.end(), 
                [](char c) { return c == '[' || c == ']' || c == ' '; }), token.end());
            
            if (!token.empty()) {
                data.push_back(std::stod(token));
            }
        }
    }
    return data;
}

// 创建数据集
void create_dataset(const std::vector<double>& data,
                   std::vector<double>& X,
                   std::vector<double>& y) {
    for (size_t i = 0; i <= data.size() - INPUT_DIM - 1; ++i) {
        for (size_t j = 0; j < INPUT_DIM; ++j) {
            X.push_back(data[i + j]);
        }
        y.push_back(data[i + INPUT_DIM]);
    }
}

// 数据归一化
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
}

// 数据反归一化
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
}

// 保存模型权重
void save_weights(const std::vector<double>& hidden_weights, 
                 const std::vector<double>& output_weights,
                 const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving weights: " << filename << std::endl;
        return;
    }

    size_t hidden_size = hidden_weights.size();
    size_t output_size = output_weights.size();
    
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(hidden_weights.data()), 
               hidden_size * sizeof(double));
    file.write(reinterpret_cast<const char*>(output_weights.data()), 
               output_size * sizeof(double));
}

// 加载模型权重
bool load_weights(std::vector<double>& hidden_weights,
                 std::vector<double>& output_weights,
                 const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for loading weights: " << filename << std::endl;
        return false;
    }

    size_t hidden_size, output_size;
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));

    if (hidden_size != hidden_weights.size() || output_size != output_weights.size()) {
        std::cerr << "Weight dimensions mismatch!" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(hidden_weights.data()), 
              hidden_size * sizeof(double));
    file.read(reinterpret_cast<char*>(output_weights.data()), 
              output_size * sizeof(double));
    return true;
}

int main() {
    // 1. 加载数据
    auto data = load_json_bandwidth("starlink_bw.json");
    if (data.empty()) {
        std::cerr << "Failed to load data" << std::endl;
        return -1;
    }

    // 2. 数据归一化
    double min_val, max_val;
    normalize_data(data, min_val, max_val);

    // 3. 创建训练集
    std::vector<double> X, y;
    create_dataset(data, X, y);

    // 4. 划分训练集和测试集
    size_t total_samples = X.size() / INPUT_DIM;
    size_t train_size = total_samples * 0.8;
    
    std::vector<double> X_train(X.begin(), X.begin() + train_size * INPUT_DIM);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<double> X_test(X.begin() + train_size * INPUT_DIM, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    // 5. 初始化模型参数
    std::vector<double> h_hidden_weights(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> h_output_weights(HIDDEN_DIM * OUTPUT_DIM);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> hidden_dist(0.0, sqrt(2.0 / INPUT_DIM));
    std::normal_distribution<double> output_dist(0.0, sqrt(2.0 / HIDDEN_DIM));
    
    for(auto& w : h_hidden_weights) w = hidden_dist(gen);
    for(auto& w : h_output_weights) w = output_dist(gen);

    // 6. 分配设备内存
    double *d_X, *d_y, *d_hidden_weights, *d_output_weights;
    double *d_hidden_output, *d_final_output;
    double *d_output_grad, *d_hidden_grad;
    double *d_hidden_weight_grad, *d_output_weight_grad;
    
    hipMalloc(&d_X, BATCH_SIZE * INPUT_DIM * sizeof(double));
    hipMalloc(&d_y, BATCH_SIZE * sizeof(double));
    hipMalloc(&d_hidden_weights, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_output_weights, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden_output, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_final_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_hidden_weight_grad, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_output_weight_grad, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));

    // 拷贝初始权重到设备
    hipMemcpy(d_hidden_weights, h_hidden_weights.data(), 
              h_hidden_weights.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_output_weights, h_output_weights.data(), 
              h_output_weights.size() * sizeof(double), hipMemcpyHostToDevice);

    // 7. 训练循环
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double total_loss = 0.0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < train_size; i += BATCH_SIZE) {
            size_t batch_size = std::min(static_cast<size_t>(BATCH_SIZE), train_size - i);

            // 加载批次数据
            hipMemcpy(d_X, &X_train[i * INPUT_DIM], 
                     batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y, &y_train[i], 
                     batch_size * sizeof(double), hipMemcpyHostToDevice);

            // 前向传播
            dim3 block(16, 16);
            dim3 grid((HIDDEN_DIM + block.x - 1) / block.x, 
                     (batch_size + block.y - 1) / block.y);

            // 隐藏层
            matmul<<<grid, block>>>(d_X, d_hidden_weights, d_hidden_output, 
                                  batch_size, HIDDEN_DIM, INPUT_DIM);
            relu_forward<<<(batch_size * HIDDEN_DIM + 255) / 256, 256>>>
                (d_hidden_output, batch_size * HIDDEN_DIM);

            // 输出层
            matmul<<<grid, block>>>(d_hidden_output, d_output_weights, d_final_output,
                                  batch_size, OUTPUT_DIM, HIDDEN_DIM);

            // 计算损失和梯度
            double* d_loss;
            hipMalloc(&d_loss, sizeof(double));
            hipMemset(d_loss, 0, sizeof(double));
            
            compute_mse_loss<<<(batch_size + 255) / 256, 256>>>
                (d_final_output, d_y, d_loss, d_output_grad, batch_size);

            // 反向传播
            double* d_output_weights_T;
            hipMalloc(&d_output_weights_T, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
            
            transpose<<<grid, block>>>(d_output_weights, d_output_weights_T, 
                                     HIDDEN_DIM, OUTPUT_DIM);
            matmul<<<grid, block>>>(d_output_grad, d_output_weights_T, d_hidden_grad,
                                  batch_size, HIDDEN_DIM, OUTPUT_DIM);
            compute_relu_backward<<<(batch_size * HIDDEN_DIM + 255) / 256, 256>>>
                (d_hidden_grad, d_hidden_output, batch_size * HIDDEN_DIM);

            // 计算权重梯度
            double* d_X_T, *d_hidden_output_T;
            hipMalloc(&d_X_T, INPUT_DIM * batch_size * sizeof(double));
            hipMalloc(&d_hidden_output_T, HIDDEN_DIM * batch_size * sizeof(double));
            
            transpose<<<grid, block>>>(d_X, d_X_T, batch_size, INPUT_DIM);
            transpose<<<grid, block>>>(d_hidden_output, d_hidden_output_T, 
                                     batch_size, HIDDEN_DIM);
                                     
            matmul<<<grid, block>>>(d_X_T, d_hidden_grad, d_hidden_weight_grad,
                                  INPUT_DIM, HIDDEN_DIM, batch_size);
            matmul<<<grid, block>>>(d_hidden_output_T, d_output_grad, d_output_weight_grad,
                                  HIDDEN_DIM, OUTPUT_DIM, batch_size);

            // 更新权重
            sgd_update<<<(INPUT_DIM * HIDDEN_DIM + 255) / 256, 256>>>
                (d_hidden_weights, d_hidden_weight_grad, LEARNING_RATE, 
                 INPUT_DIM * HIDDEN_DIM);
            sgd_update<<<(HIDDEN_DIM * OUTPUT_DIM + 255) / 256, 256>>>
                (d_output_weights, d_output_weight_grad, LEARNING_RATE,
                 HIDDEN_DIM * OUTPUT_DIM);

            // 记录损失
            double batch_loss;
            hipMemcpy(&batch_loss, d_loss, sizeof(double), hipMemcpyDeviceToHost);
            total_loss += batch_loss;

            // 清理临时内存
            hipFree(d_loss);
            hipFree(d_output_weights_T);
            hipFree(d_X_T);
            hipFree(d_hidden_output_T);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if ((epoch + 1) % 10 == 0) {
            std::cout << "[Epoch " << std::setw(3) << epoch + 1 << "] "
                     << "Loss: " << std::scientific << std::setprecision(4) << total_loss
                     << ", Time: " << duration.count() << " ms" << std::endl;
        }
    }

    // 8. 预测
    std::vector<double> predictions;
    double *d_test_X, *d_pred;
    hipMalloc(&d_test_X, INPUT_DIM * sizeof(double));
    hipMalloc(&d_pred, OUTPUT_DIM * sizeof(double));
    
    std::cout << "\nGenerating predictions..." << std::endl;

    for (size_t i = 0; i < y_test.size(); ++i) {
        hipMemcpy(d_test_X, &X_test[i * INPUT_DIM], 
                 INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        
        dim3 block(16, 16);
        dim3 grid((HIDDEN_DIM + block.x - 1) / block.x, 1);
        
        matmul<<<grid, block>>>(d_test_X, d_hidden_weights, d_hidden_output, 
                               1, HIDDEN_DIM, INPUT_DIM);
        relu_forward<<<(HIDDEN_DIM + 255) / 256, 256>>>(d_hidden_output, HIDDEN_DIM);
        matmul<<<grid, block>>>(d_hidden_output, d_output_weights, d_pred,
                               1, OUTPUT_DIM, HIDDEN_DIM);

        double pred;
        hipMemcpy(&pred, d_pred, sizeof(double), hipMemcpyDeviceToHost);
        predictions.push_back(pred);
    }

    // 9. 反归一化预测结果
    denormalize_data(predictions, min_val, max_val);
    std::vector<double> actual_y = y_test;
    denormalize_data(actual_y, min_val, max_val);

    // 10. 计算和输出预测结果
    double mse_error = 0.0;
    std::cout << "\nPrediction Results (first 10 samples):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    for (size_t i = 0; i < std::min(size_t(10), predictions.size()); ++i) {
        std::cout << "Step " << std::setw(2) << i << ": Predicted = " 
                 << std::setw(10) << predictions[i] 
                 << ", Actual = " << std::setw(10) << actual_y[i] << std::endl;
        mse_error += pow(predictions[i] - actual_y[i], 2);
    }
    mse_error /= predictions.size();
    std::cout << "\nMean Squared Error: " << mse_error << std::endl;

    // 11. 保存模型权重
    hipMemcpy(h_hidden_weights.data(), d_hidden_weights,
              h_hidden_weights.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_output_weights.data(), d_output_weights,
              h_output_weights.size() * sizeof(double), hipMemcpyDeviceToHost);
    
    save_weights(h_hidden_weights, h_output_weights, "mlp_weights.bin");

    // 12. 清理设备内存
    hipFree(d_X);
    hipFree(d_y);
    hipFree(d_hidden_weights);
    hipFree(d_output_weights);
    hipFree(d_hidden_output);
    hipFree(d_final_output);
    hipFree(d_output_grad);
    hipFree(d_hidden_grad);
    hipFree(d_hidden_weight_grad);
    hipFree(d_output_weight_grad);
    hipFree(d_test_X);
    hipFree(d_pred);

    return 0;
}