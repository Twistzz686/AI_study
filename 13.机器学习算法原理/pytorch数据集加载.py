import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==================== 1. 数据准备 ====================
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]

data = np.array(data)
x_data = data[:, 0]  # 提取x特征
y_data = data[:, 1]  # 提取y标签

# ==================== 2. 转换为PyTorch张量 ====================
x_train = torch.tensor(x_data, dtype=torch.float32)  # 形状 (10,)
y_train = torch.tensor(y_data, dtype=torch.float32)  # 形状 (10,)

dataset = TensorDataset(x_train, y_train)  # 创建数据集

# ==================== 3. 设置随机种子 ====================
seed = 42
torch.manual_seed(seed)  # 固定随机种子，保证可复现


# ==================== 4. 创建模型 ====================
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 单层线性层: y = wx + b

    def forward(self, x):
        x = self.linear(x)
        return x


model = LinearModel()

# ==================== 5. 定义损失函数和优化器 ====================
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# ==================== 6. 训练循环 ====================
epoches = 500
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)  # 批次加载，随机打乱

for n in range(1, epoches + 1):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        y_hat = model(batch_x.unsqueeze(1))  # 前向传播，(5,) → (5,1)
        loss = criterion(y_hat.squeeze(1), batch_y)  # 计算损失，(5,1) → (5,)
        total_loss += loss

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    avg_loss = total_loss / len(dataloader)  # 平均损失

    if n % 10 == 0 or n == 1:
        print(f"epoches:{n}, avg_loss:{avg_loss}")  # 每10轮输出一次