import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==================== 1. 数据加载 ====================
# 读取 Excel 数据文件
data = pd.read_excel('./Real estate valuation data set.xlsx')
print(data.head())  # 查看前5行数据

# ==================== 2. 数据预处理 ====================
# 对分类特征进行独热编码（将便利店数量转换为虚拟变量）
# 原列 'X4 number of convenience stores' 会被替换为多列如 _0, _1, _2...
data = pd.get_dummies(data, columns=['X4 number of convenience stores'])
print(data.keys())  # 查看所有列名

# ==================== 3. 划分特征和标签 ====================
# 选择特征列（注意：这里错误地把 Y 也包含在 X 中了！）
# 应排除 'Y house price of unit area'，否则会造成数据泄露
X = data[['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station', 'X5 latitude', 'X6 longitude','X4 number of convenience stores_0',
       'X4 number of convenience stores_1',
       'X4 number of convenience stores_2',
       'X4 number of convenience stores_3',
       'X4 number of convenience stores_4',
       'X4 number of convenience stores_5',
       'X4 number of convenience stores_6',
       'X4 number of convenience stores_7',
       'X4 number of convenience stores_8',
       'X4 number of convenience stores_9',
       'X4 number of convenience stores_10']]

# 标签（房价）
Y = data['Y house price of unit area']

# ==================== 4. 划分训练集和测试集 ====================
# test_size=0.2: 20% 数据作为测试集，80% 作为训练集
# random_state=42: 固定随机种子，确保结果可复现
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==================== 5. 特征标准化 ====================
# 创建标准化器（将数据转换为均值0、标准差1）
scaler = StandardScaler()
# 训练集：计算均值和标准差，并转换
X_train_scaled = scaler.fit_transform(X_train)
# 测试集：使用训练集的参数转换（不重新计算）
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)  # 查看标准化后的数据

# ==================== 6. 转换为 PyTorch 张量 ====================
# 特征张量（形状: [样本数, 特征数]）
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 标签张量（需要 reshape 为 [样本数, 1] 以便与模型输出匹配）
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)


# ==================== 7. 定义线性回归模型 ====================
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 全连接层：输入 input_size 维，输出 1 维
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播：x @ weight.T + bias
        return self.linear(x)


# 创建模型实例，输入特征数为 X_train 的列数
model = LinearRegressionModel(X_train.shape[1])

# ==================== 8. 定义损失函数和优化器 ====================
criterion = nn.MSELoss()  # 均方误差损失（回归任务常用）
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam 优化器，学习率 0.1

# ==================== 9. 训练循环 ====================
num_epochs = 500  # 训练轮数

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式（影响 Dropout、BN 等层）

    optimizer.zero_grad()  # 清空之前的梯度（避免累积）

    # 前向传播：计算预测值
    output = model(X_train_tensor)

    # 计算损失
    loss = criterion(output, Y_train_tensor)

    # 反向传播：计算梯度
    loss.backward()

    # 更新参数：执行梯度下降
    optimizer.step()

    # 每 100 个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# ==================== 10. 模型评估 ====================
model.eval()  # 设置为评估模式（关闭 Dropout、BN 使用全局统计）

with torch.no_grad():  # 禁用梯度计算（节省内存和计算）
    # 在测试集上预测
    predictions = model(X_test_tensor)
    # 计算测试损失
    test_loss = criterion(predictions, Y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# ==================== 11. 可视化预测结果 ====================
# 将张量转换为 numpy 数组（便于绘图）
predictions = predictions.numpy()
Y_test_numpy = Y_test_tensor.numpy()

# 创建散点图：真实值 vs 预测值
plt.figure(0)
plt.scatter(Y_test_numpy, predictions, color='blue', alpha=0.6, label='Predictions')
# 绘制理想线（y = x），表示完美预测
plt.plot([min(Y_test_numpy), max(Y_test_numpy)],
         [min(Y_test_numpy), max(Y_test_numpy)],
         linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()

# ==================== 注意事项 ====================
# ⚠️ 代码中的问题：
# 1. X 包含了 Y 列（数据泄露），应排除 'Y house price of unit area'
# 2. 正确写法：X = data.drop('Y house price of unit area', axis=1)