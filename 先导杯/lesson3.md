# 基于MLP的低轨卫星网络带宽预测性能优化报告

## 1. 任务概述

本任务旨在设计并实现一个基于多层感知机(MLP)的神经网络模型，用于预测低轨卫星网络的下行带宽值。主要挑战包括：

- 实现完整的MLP训练流程（前向传播、反向传播、梯度下降）
- 使用DCU加速卡进行高性能计算
- 实现准确的带宽预测并优化性能

**输入输出规格**：

- 输入：连续10个时间点的带宽值(t₀到t₉)
- 输出：下一个时间点的带宽值(t₁₀)

## 2. 数据预处理

### 2.1 数据加载与处理

```cpp
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    // 数据解析逻辑...
    return data;
}
```

- 从JSON文件加载原始带宽数据

### 2.2 滑动窗口样本构建

```cpp
void create_dataset(const std::vector<double>& data,
                   std::vector<double>& X,
                   std::vector<double>& y) {
    for (size_t i = 0; i <= data.size() - INPUT_DIM - 1; ++i) {
        for (size_t j = 0; j < INPUT_DIM; ++j) {
            X.push_back(data[i + j]); // 输入序列
        }
        y.push_back(data[i + INPUT_DIM]); // 输出目标
    }
}
```

- 窗口大小: N=10
- 每个样本: 10个连续时间点 → 预测第11个时间点
- 生成样本总数: 数据长度 - 10

### 2.3 数据归一化

```cpp
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val); // 归一化到[0,1]
    }
}
```

- 归一化方法: Min-Max归一化
- 目的: 消除量纲影响，加速模型收敛
- 存储min_val/max_val用于后续反归一化

### 2.4 数据集划分

- 训练集: 80% (前80%样本)
- 测试集: 20% (后20%样本)
- 批处理大小: 256样本/批次

## 3. 模型设计

### 3.1 MLP架构

| 层类型 | 神经元数量 | 激活函数 | 参数数量  |
| ------ | ---------- | -------- | --------- |
| 输入层 | 10         | -        | -         |
| 隐藏层 | 32         | ReLU     | 10×32=320 |
| 输出层 | 1          | 线性     | 32×1=32   |

**总参数**: 352

### 3.2 关键组件

1. **前向传播**:

   ```cpp
   // 隐藏层计算
   matmul<<<grid, block>>>(d_X, d_hidden_weights, d_hidden_output, 
                         batch_size, HIDDEN_DIM, INPUT_DIM);
   relu_forward<<<...>>>(d_hidden_output, batch_size * HIDDEN_DIM);
   
   // 输出层计算
   matmul<<<grid, block>>>(d_hidden_output, d_output_weights, d_final_output,
                         batch_size, OUTPUT_DIM, HIDDEN_DIM);
   ```

2. **损失函数**: 均方误差(MSE)

   ```cpp
   compute_mse_loss<<<...>>>(d_final_output, d_y, d_loss, d_output_grad, batch_size);
   ```

3. **反向传播**:

   ```cpp
   // 输出层梯度
   matmul<<<...>>>(d_output_grad, d_output_weights_T, d_hidden_grad,
                 batch_size, HIDDEN_DIM, OUTPUT_DIM);
   
   // ReLU梯度
   compute_relu_backward<<<...>>>(d_hidden_grad, d_hidden_output, ...);
   
   // 权重梯度计算
   matmul<<<...>>>(d_X_T, d_hidden_grad, d_hidden_weight_grad,
                 INPUT_DIM, HIDDEN_DIM, batch_size);
   ```

4. **参数更新** (SGD):

   ```cpp
   sgd_update<<<...>>>(d_hidden_weights, d_hidden_weight_grad, 
                     LEARNING_RATE, INPUT_DIM * HIDDEN_DIM);
   ```

### 3.3 训练参数

| 参数       | 值       | 说明               |
| ---------- | -------- | ------------------ |
| 学习率     | 1e-4     | 梯度下降步长       |
| 批次大小   | 256      | 每次训练样本数     |
| 训练轮数   | 200      | 完整遍历数据集次数 |
| 权重初始化 | He初始化 | 适配ReLU激活函数   |

## 4. DCU加速实现

### 4.1 计算优化策略

1. **核函数设计**:

   - 矩阵乘法(matmul)
   - ReLU激活(relu_forward)
   - ReLU梯度(compute_relu_backward)
   - 权重更新(sgd_update)

2. **并行计算**:

   ```cpp
   dim3 block(16, 16); // 256线程/块
   dim3 grid((HIDDEN_DIM + 15)/16, (batch_size + 15)/16); // 动态网格大小
   ```

3. **内存管理**:

   - 预分配所有设备内存
   - 使用批处理减少主机-设备通信
   - 及时释放临时内存

### 4.2 训练流程优化

```cpp
for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    // 1. 数据批量加载到DCU
    // 2. 前向传播（矩阵乘法+激活）
    // 3. 损失计算
    // 4. 反向传播（梯度计算）
    // 5. 权重更新
    // 6. 性能监控
}
```

## 5. 性能评测

![img_v3_02mt_6f1d6cbd-0bc2-4cfa-8b35-2df2ddc0e26g](D:\许洋计算机科学与技术\计算机组成原理\先导杯\pic\img_v3_02mt_6f1d6cbd-0bc2-4cfa-8b35-2df2ddc0e26g.jpg)

## 6. 优化挑战与解决方案

1. **内存瓶颈**:
   - 问题: 频繁分配释放临时内存
   - 优化: 预分配所有GPU内存并复用
2. **计算效率**:
   - 问题: 小矩阵乘法效率低
   - 优化: 使用合并内存访问，增大线程块(16×16)
3. **数值稳定性**:
   - 问题: 梯度爆炸风险
   - 优化: He初始化 + 学习率衰减
4. **数据流水线**:
   - 问题: 主机-设备数据传输瓶颈
   - 优化: 异步数据传输与计算重叠

## 7. 结论

本实现成功构建了一个基于MLP的低轨卫星带宽预测模型，通过DCU加速实现了显著的性能提升：

- 模型在测试集上达到17.781128的MSE，预测趋势与实际值高度一致
- 优化的HIP核函数充分利用了DCU的并行计算能力

**未来优化方向**：

1. 增加隐藏层深度提升模型容量
2. 实现学习率自适应调度
3. 引入双向数据预取机制
4. 探索混合精度训练

本方案为低轨卫星网络提供了高效的带宽预测能力，可有效支持卫星资源调度和网络优化决策。