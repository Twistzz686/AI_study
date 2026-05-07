import torch
import torch.nn as nn
import numpy as np

# ==================== 1. 数据准备 ====================
# 原始数据：每个子列表为 [x, y] 坐标对
# 从数据分布看，y 和 x 大致呈线性关系（y ≈ 50x + 30 左右）
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]

# 转换为 NumPy 数组，便于数据处理
data = np.array(data)

# 提取特征 x（第一列）和标签 y（第二列）
x_data = data[:, 0]  # x: [-0.5, 1.8, 0.9, 0.4, -1.4, -1.4, -1.8, 1.5, 0.4, 0.8]
y_data = data[:, 1]  # y: [7.7, 98.5, 57.8, 39.2, -15.7, -37.3, -49.1, 75.6, 34.0, 62.3]

# ==================== 2. 转换为 PyTorch 张量 ====================
# 将 NumPy 数组转换为 PyTorch 张量，数据类型为 float32
# 注意：此时 x_train 和 y_train 是一维张量，形状为 (10,)
x_train = torch.tensor(x_data, dtype=torch.float32)
y_train = torch.tensor(y_data, dtype=torch.float32)

# ==================== 3. 设置随机种子 ====================
# 固定随机种子，确保实验结果可复现
# 这会影响模型参数的初始值
seed = 42
torch.manual_seed(seed)

# ==================== 4. 创建模型 ====================
# nn.Linear(1, 1) 创建一个线性层，实现 y = w * x + b
# 参数含义：输入特征数=1，输出特征数=1
# 模型内部会自动创建可训练参数：weight(形状1x1) 和 bias(形状1)
model = nn.Linear(1, 1)

# nn.Sequential是pytorch的一个模块容器，按顺序组合多个网络层
# 默认带forward方法，会定义模型的向前传播逻辑，给定输入，经过逻辑，得到输出
# modef = nn.Sequential(nn.Linear(1,1))
# ----------------

# nn.ModuleList,如果要使用，需要自己定构造类
# model = nn.ModuleList([nn.Linear(1,1)])
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.layers = nn.ModuleList([nn.Linear(1,1)])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
# model = LinearModel()
# ----------------------


# nn.ModuleDict
# model = nn.ModuleDict({"linear":nn.Linear(1, 1)})

# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         layers = nn.ModuleDict({"linear": nn.Linear(1, 1)})
#
#     def forward(self, x):
#         for layer in self.layers.values:
#             x = layer(x)
#         return x

# ------------------------

# 实际上最常用的
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(1,1)
#         self.linear2 = nn.Linear(2,1)
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.linear2(x)
#         return x
# model = LinearModel()


# ==================== 5. 定义损失函数和优化器 ====================
# MSELoss: 均方误差损失，适用于回归任务
# 公式: loss = mean((y_pred - y_true)^2)
criterion = nn.MSELoss()

# SGD: 随机梯度下降优化器
# model.parameters(): 传入模型的所有可训练参数（weight 和 bias）
# lr=0.01: 学习率，控制参数更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ==================== 6. 训练循环 ====================
epoches = 500  # 训练轮数

for n in range(1, epoches + 1):
    # ----- 前向传播 -----
    # x_train.unsqueeze(1): 将 x_train 从 (10,) 变成 (10, 1)
    # 因为 Linear 层期望输入形状为 (batch_size, in_features)
    # unsqueeze(1) 在索引1处添加维度，将一维列向量变成二维列向量
    y_hat = model(x_train.unsqueeze(1))  # 预测值，形状 (10, 1)

    # 计算损失
    # y_hat 形状是 (10, 1)，y_train 形状是 (10,)
    # 需要对 y_hat 也做 unsqueeze 确保形状一致
    # 或者 y_train.unsqueeze(1) 使其变为 (10, 1)
    loss = criterion(y_hat, y_train.unsqueeze(1))  # 注意：这里对 y_train 添加了维度

    # ----- 反向传播 -----
    # 清空之前的梯度（梯度默认会累加）
    optimizer.zero_grad()

    # 反向传播，计算损失对模型参数的梯度
    loss.backward()

    # 更新模型参数：param = param - lr * gradient
    optimizer.step()

    # ----- 输出训练日志 -----
    # 每 10 个 epoch 输出一次，同时输出第 1 个 epoch
    if n % 10 == 0 or n == 1:
        print(f"epoches:{n}, loss:{loss.item():.6f}")  # .item() 提取标量值

# ==================== 7. 训练结果说明 ====================
# 训练完成后，可以查看学习到的参数：
# print(f"学习到的权重 w: {model.weight.item():.4f}")
# print(f"学习到的偏置 b: {model.bias.item():.4f}")
#
# 预期结果：模型应该学习到近似 y = 50x + 35 的线性关系