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

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 超参数定义
// 修改超参数配置
#define INPUT_DIM 10
#define HIDDEN_DIM 64        // 增大隐藏层 
#define OUTPUT_DIM 1
#define BATCH_SIZE 32        // 保持较小的批量大小
#define EPOCHS 2000          // 保持较大的训练轮数
#define BASE_LR 1e-4         // 调整基础学习率
#define WEIGHT_DECAY 1e-5    // 调整正则化系数
#define MOMENTUM 0.9
#define PATIENCE 100         // 增加早停耐心值
// // 预定义参数，可根据需求修改
// #define INPUT_DIM 10        // 输入维度保持不变
// #define HIDDEN_DIM 64      // 增大隐藏层
// #define OUTPUT_DIM 1        // 输出维度保持不变
// #define BATCH_SIZE 32       // 减小批量
// #define EPOCHS 1000          // 增加轮数
// #define LEARNING_RATE 5e-4  // 增大学习率
// #define WEIGHT_DECAY 1e-4   // 减小正则化
// #define PATIENCE 50         // 增加早停耐心

struct OptimParams {
    double *v_W1, *v_W2, *v_b1, *v_b2;  // 速度矩阵
    
    void allocate() {
        hipMalloc(&v_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
        hipMalloc(&v_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
        hipMalloc(&v_b1, HIDDEN_DIM * sizeof(double));
        hipMalloc(&v_b2, OUTPUT_DIM * sizeof(double));
        
        // 初始化为0
        hipMemset(v_W1, 0, INPUT_DIM * HIDDEN_DIM * sizeof(double));
        hipMemset(v_W2, 0, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
        hipMemset(v_b1, 0, HIDDEN_DIM * sizeof(double));
        hipMemset(v_b2, 0, OUTPUT_DIM * sizeof(double));
    }
    
    void free() {
        hipFree(v_W1); hipFree(v_W2);
        hipFree(v_b1); hipFree(v_b2);
    }
};

__global__ void sgd_momentum_update(double* weights, double* velocity,
                                  const double* grad, double lr, double momentum,
                                  double weight_decay, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        velocity[idx] = momentum * velocity[idx] - lr * (grad[idx] + weight_decay * weights[idx]);
        weights[idx] += velocity[idx];
    }
}

void shuffle_data(std::vector<double>& X, std::vector<double>& y) {
    std::vector<int> indices(y.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    std::vector<double> X_shuffled(X.size());
    std::vector<double> y_shuffled(y.size());
    
    for(size_t i = 0; i < indices.size(); i++) {
        for(int j = 0; j < INPUT_DIM; j++) {
            X_shuffled[i * INPUT_DIM + j] = X[indices[i] * INPUT_DIM + j];
        }
        y_shuffled[i] = y[indices[i]];
    }
    
    X = X_shuffled;
    y = y_shuffled;
}

struct MLPParams {
    double *W1, *W2, *b1, *b2;
    double *dW1, *dW2, *db1, *db2;
    double *h1, *h1_relu, *output;
    double *bn1_mean, *bn1_var;    // 批量归一化参数
    double *bn1_gamma, *bn1_beta;
    
    void allocate() {
        hipMalloc(&W1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
        hipMalloc(&W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
        hipMalloc(&b1, HIDDEN_DIM * sizeof(double));
        hipMalloc(&b2, OUTPUT_DIM * sizeof(double));
        
        hipMalloc(&dW1, INPUT_DIM * HIDDEN_DIM * sizeof(double));
        hipMalloc(&dW2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
        hipMalloc(&db1, HIDDEN_DIM * sizeof(double));
        hipMalloc(&db2, OUTPUT_DIM * sizeof(double));
        
        hipMalloc(&h1, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
        hipMalloc(&h1_relu, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
        hipMalloc(&output, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
        
        // 分配批量归一化参数内存
        hipMalloc(&bn1_mean, HIDDEN_DIM * sizeof(double));
        hipMalloc(&bn1_var, HIDDEN_DIM * sizeof(double));
        hipMalloc(&bn1_gamma, HIDDEN_DIM * sizeof(double));
        hipMalloc(&bn1_beta, HIDDEN_DIM * sizeof(double));
        
        // 初始化批量归一化参数
        std::vector<double> h_gamma(HIDDEN_DIM, 1.0);
        std::vector<double> h_beta(HIDDEN_DIM, 0.0);
        hipMemcpy(bn1_gamma, h_gamma.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(bn1_beta, h_beta.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice);
    }
    
    void free() {
        hipFree(W1); hipFree(W2); hipFree(b1); hipFree(b2);
        hipFree(dW1); hipFree(dW2); hipFree(db1); hipFree(db2);
        hipFree(h1); hipFree(h1_relu); hipFree(output);
        hipFree(bn1_mean); hipFree(bn1_var);
        hipFree(bn1_gamma); hipFree(bn1_beta);
    }
};
// 参数初始化
void initialize_params(MLPParams& params) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> w1_dist(0.0, sqrt(2.0 / INPUT_DIM));
    std::normal_distribution<double> w2_dist(0.0, sqrt(2.0 / HIDDEN_DIM));
    
    std::vector<double> h_W1(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> h_W2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> h_b1(HIDDEN_DIM, 0.0);
    std::vector<double> h_b2(OUTPUT_DIM, 0.0);
    
    // Xavier初始化
    for(size_t i = 0; i < h_W1.size(); i++) {
        h_W1[i] = w1_dist(gen);
    }
    for(size_t i = 0; i < h_W2.size(); i++) {
        h_W2[i] = w2_dist(gen);
    }
    
    hipMemcpy(params.W1, h_W1.data(), h_W1.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(params.W2, h_W2.data(), h_W2.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(params.b1, h_b1.data(), h_b1.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(params.b2, h_b2.data(), h_b2.size() * sizeof(double), hipMemcpyHostToDevice);
}

// 以下函数和main函数均不为固定形式，可自行按照需求修改

// HIP kernels函数形式，需要自行设计
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transpose(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_forward(double* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = max(0.0, input[idx]);
    }
}


__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (pred[idx] - target[idx]) / size; // 增加批量大小归一化
    }
}

__global__ void compute_relu_backward(double* grad, const double* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 如果前向传播时的输入小于等于0，梯度应该为0
        // 否则保持梯度不变(相当于乘以1)
        if (pre_activation[idx] <= 0) {
            grad[idx] = 0.0;
        }
    }
}

__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        loss[idx] = 0.5f * (pred[idx] - target[idx]) * (pred[idx] - target[idx]);
    }
}

__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx]; // SGD更新规则
    }
}

__global__ void add_l2_regularization(double* grad, const double* weights, double lambda, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] += lambda * weights[idx];
    }
}

__global__ void batch_norm_forward(double* input, const double* gamma, const double* beta,
                                 double* mean, double* var, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        double sum = 0.0;
        double sq_sum = 0.0;
        
        // 计算均值和方差
        for(int n = 0; n < N; n++) {
            double val = input[n * C + c];
            sum += val;
            sq_sum += val * val;
        }
        
        mean[c] = sum / N;
        var[c] = sq_sum / N - mean[c] * mean[c] + 1e-5;
        
        // 归一化和缩放
        for(int n = 0; n < N; n++) {
            double normalized = (input[n * C + c] - mean[c]) / sqrt(var[c]);
            input[n * C + c] = gamma[c] * normalized + beta[c];
        }
    }
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> bandwidth_data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return bandwidth_data;
    }
    
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string token;
    
    // 跳过开头的 '['
    ss >> token;
    
    while (std::getline(ss, token, ',')) {
        // 清理数字字符串
        token.erase(std::remove_if(token.begin(), token.end(), 
            [](char c) { return c == ' ' || c == '[' || c == ']'; }), token.end());
        
        if (!token.empty()) {
            bandwidth_data.push_back(std::stod(token));
        }
    }
    
    return bandwidth_data;
}

// 创建数据集
// 新的数据集创建和划分函数
void create_and_split_dataset(const std::vector<double>& data,
                            std::vector<double>& X_train, std::vector<double>& y_train,
                            std::vector<double>& X_test, std::vector<double>& y_test,
                            double train_ratio = 0.8) {
    // 计算总样本数
    int total_samples = data.size() - INPUT_DIM;
    int train_size = static_cast<int>(total_samples * train_ratio);
    int test_size = total_samples - train_size;
    
    // 创建索引并随机打乱
    std::vector<int> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // 调整数组大小
    X_train.resize(train_size * INPUT_DIM);
    y_train.resize(train_size);
    X_test.resize(test_size * INPUT_DIM);
    y_test.resize(test_size);
    
    // 划分训练集
    for (int i = 0; i < train_size; ++i) {
        int idx = indices[i];
        for (int j = 0; j < INPUT_DIM; ++j) {
            X_train[i * INPUT_DIM + j] = data[idx + j];
        }
        y_train[i] = data[idx + INPUT_DIM];
    }
    
    // 划分测试集
    for (int i = 0; i < test_size; ++i) {
        int idx = indices[i + train_size];
        for (int j = 0; j < INPUT_DIM; ++j) {
            X_test[i * INPUT_DIM + j] = data[idx + j];
        }
        y_test[i] = data[idx + INPUT_DIM];
    }
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    return;
}

// ----------------------------- Main -------------------------------
int main() {
	// 读取带宽json文件，并生成测试集和训练集

    // 数据加载和预处理
    std::vector<double> raw_data = load_json_bandwidth("starlink_bw.json");
    double min_val, max_val;
    normalize_data(raw_data, min_val, max_val);
    
    // 创建并划分数据集
    std::vector<double> X_train, y_train, X_test, y_test;
    create_and_split_dataset(raw_data, X_train, y_train, X_test, y_test);
    
    // 初始化模型参数
    MLPParams params;
    params.allocate();
    initialize_params(params);
    
    // 初始化优化器
    OptimParams optim;
    optim.allocate();
    
    // 准备训练数据
    double *d_X_train, *d_y_train;
    hipMalloc(&d_X_train, X_train.size() * sizeof(double));
    hipMalloc(&d_y_train, y_train.size() * sizeof(double));
    hipMemcpy(d_X_train, X_train.data(), X_train.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(double), hipMemcpyHostToDevice);
    
    // 设置执行配置
    dim3 block(16, 16);
    dim3 grid((BATCH_SIZE + block.x - 1) / block.x, 
              (HIDDEN_DIM + block.y - 1) / block.y);
    
    // 训练循环
    double best_loss = INFINITY;
    int patience_counter = 0;
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 每轮开始时打乱训练数据
        shuffle_data(X_train, y_train);
        hipMemcpy(d_X_train, X_train.data(), X_train.size() * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_y_train, y_train.data(), y_train.size() * sizeof(double), hipMemcpyHostToDevice);
        
        // 使用学习率衰减
        double current_lr = BASE_LR * 0.5 * (1 + cos(M_PI * epoch / EPOCHS));
        
        double total_loss = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        
        // 批量训练
        for (int batch = 0; batch < X_train.size() / BATCH_SIZE; batch++) {
            int offset = batch * BATCH_SIZE;
            
            // 前向传播
            matmul<<<grid, block>>>(d_X_train + offset, params.W1, params.h1, 
                                BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);

            // 添加批量归一化
            batch_norm_forward<<<grid, block>>>(params.h1, params.bn1_gamma, params.bn1_beta,
                                            params.bn1_mean, params.bn1_var,
                                            BATCH_SIZE, HIDDEN_DIM);

            relu_forward<<<grid, block>>>(params.h1, BATCH_SIZE * HIDDEN_DIM);
            hipMemcpy(params.h1_relu, params.h1, 
                    BATCH_SIZE * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice);
            matmul<<<grid, block>>>(params.h1_relu, params.W2, params.output, 
                                BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);
            // 计算损失
            double* d_loss;
            hipMalloc(&d_loss, BATCH_SIZE * sizeof(double));
            compute_mse_loss<<<grid, block>>>(params.output, d_y_train + offset, 
                                            d_loss, BATCH_SIZE);
            
            std::vector<double> h_loss(BATCH_SIZE);
            hipMemcpy(h_loss.data(), d_loss, BATCH_SIZE * sizeof(double), 
                     hipMemcpyDeviceToHost);
            for(double loss : h_loss) {
                total_loss += loss;
            }
            
            // 反向传播
            compute_output_grad<<<grid, block>>>(params.output, d_y_train + offset, 
                                               params.db2, BATCH_SIZE);
            
            double* d_hidden_grad;
            hipMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
            
            matmul<<<grid, block>>>(params.db2, params.W2, d_hidden_grad, 
                                  BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);
            
            compute_relu_backward<<<grid, block>>>(d_hidden_grad, params.h1, 
                                                 BATCH_SIZE * HIDDEN_DIM);
            
            matmul_transpose<<<grid, block>>>(d_X_train + offset, d_hidden_grad, 
                                            params.dW1, INPUT_DIM, HIDDEN_DIM, BATCH_SIZE);
            
            // 添加L2正则化
            add_l2_regularization<<<grid, block>>>(params.dW1, params.W1, WEIGHT_DECAY, 
                                                 INPUT_DIM * HIDDEN_DIM);
            add_l2_regularization<<<grid, block>>>(params.dW2, params.W2, WEIGHT_DECAY, 
                                                 HIDDEN_DIM * OUTPUT_DIM);
            
            // 使用动量SGD更新
            sgd_momentum_update<<<grid, block>>>(params.W1, optim.v_W1, params.dW1,
                                               current_lr, MOMENTUM, WEIGHT_DECAY,
                                               INPUT_DIM * HIDDEN_DIM);
            sgd_momentum_update<<<grid, block>>>(params.W2, optim.v_W2, params.dW2,
                                               current_lr, MOMENTUM, WEIGHT_DECAY,
                                               HIDDEN_DIM * OUTPUT_DIM);
            sgd_momentum_update<<<grid, block>>>(params.b1, optim.v_b1, params.db1,
                                               current_lr, MOMENTUM, 0.0,
                                               HIDDEN_DIM);
            sgd_momentum_update<<<grid, block>>>(params.b2, optim.v_b2, params.db2,
                                               current_lr, MOMENTUM, 0.0,
                                               OUTPUT_DIM);
            
            hipFree(d_loss);
            hipFree(d_hidden_grad);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double current_loss = total_loss / (X_train.size() / BATCH_SIZE);
        std::cout << "[Epoch " << epoch + 1 << "/" << EPOCHS 
                  << "] Loss: " << current_loss
                  << " lr: " << current_lr 
                  << ", Time: " << duration.count() << " ms" << std::endl;
        
        // 早停检查
        if (current_loss < best_loss) {
            best_loss = current_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= PATIENCE) {
                std::cout << "Early stopping triggered at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }
    
    // 在测试集上评估
    std::vector<double> predictions(X_test.size() / INPUT_DIM);
    double *d_X_test, *d_pred;
    hipMalloc(&d_X_test, X_test.size() * sizeof(double));
    hipMalloc(&d_pred, predictions.size() * sizeof(double));
    
    hipMemcpy(d_X_test, X_test.data(), X_test.size() * sizeof(double), hipMemcpyHostToDevice);
    
    // 前向传播进行预测
    matmul<<<grid, block>>>(d_X_test, params.W1, params.h1, 
                           predictions.size(), HIDDEN_DIM, INPUT_DIM);
    relu_forward<<<grid, block>>>(params.h1, predictions.size() * HIDDEN_DIM);
    matmul<<<grid, block>>>(params.h1, params.W2, d_pred, 
                           predictions.size(), OUTPUT_DIM, HIDDEN_DIM);
    
    hipMemcpy(predictions.data(), d_pred, predictions.size() * sizeof(double), 
              hipMemcpyDeviceToHost);
    
    // 反归一化预测结果
    denormalize_data(predictions, min_val, max_val);
    
    // 计算并打印测试集MSE
    double test_mse = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Predicted: " << predictions[i] 
                  << ", Actual: " << y_test[i] << std::endl;
        test_mse += (predictions[i] - y_test[i]) * (predictions[i] - y_test[i]);
    }
    test_mse /= predictions.size();
    std::cout << "Test MSE: " << test_mse << std::endl;
    
    // 清理内存
    params.free();
    optim.free();
    hipFree(d_X_train);
    hipFree(d_y_train);
    hipFree(d_X_test);
    hipFree(d_pred);
    
    return 0;
}
