# ==================== 导入必要的库 ====================
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# ==================== 1. 数据加载 ====================
# 读取垃圾邮件数据集（没有表头，使用默认列索引）
dataset = pd.read_csv('./spambase.data', header=None)

# ==================== 2. 特征与标签分离 ====================
# 特征：所有行，除最后一列外的所有列
X = dataset.iloc[:, :-1]
# 标签：所有行，最后一列（0=非垃圾邮件，1=垃圾邮件）
Y = dataset.iloc[:, -1]

# ==================== 3. 划分训练集和测试集 ====================
# test_size=0.2：20% 数据用于测试，80% 用于训练
# random_state=42：固定随机种子，保证结果可复现
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==================== 4. 特征标准化 ====================
# 创建标准化器（将数据转换为均值0、标准差1）
scaler = StandardScaler()
# 训练集：计算均值和标准差，并转换
X_train_scaled = scaler.fit_transform(X_train)
# 测试集：使用训练集的参数转换（不重新计算）
X_test_scaled = scaler.transform(X_test)

# ==================== 5. 转换为 PyTorch 张量 ====================
# 特征张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# 标签张量（注意：.values 提取 numpy 数组，float32 与模型输出匹配）
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

# ==================== 6. 创建 DataLoader ====================
# 将特征和标签打包成数据集
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# 创建数据加载器：batch_size=64（每批64个样本），shuffle=True（每个 epoch 打乱顺序）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ==================== 7. 定义逻辑回归模型 ====================
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # 线性层：输入特征数 = input_size，输出 1 维（logit）
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播：线性变换 → Sigmoid 得到概率
        return torch.sigmoid(self.linear(x))


# 获取输入特征数（从训练集张量的第二维得到）
input_size = X_train_tensor.shape[1]  # 例如：57 个特征
model = LogisticRegression()

# ==================== 8. 定义损失函数和优化器 ====================
# 二分类交叉熵损失（BCE = Binary Cross Entropy）
criterion = nn.BCELoss()
# Adam 优化器（自适应学习率），学习率 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==================== 9. 训练循环 ====================
num_epochs = 100  # 训练轮数

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式（影响 Dropout、BatchNorm 等层）
    total_loss = 0  # 累计当前 epoch 的总损失

    # 遍历每个 batch
    for inputs, labels in train_loader:
        # ----- 前向传播 -----
        outputs = model(inputs)  # 预测概率，形状 (batch_size, 1)
        loss = criterion(outputs, labels.view(-1, 1))  # 计算损失，labels 需要 reshape 为 (batch,1)

        # ----- 反向传播 -----
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 累计损失（.item() 提取标量值）
        total_loss += loss.item()

    # 计算当前 epoch 的平均损失
    avg_loss = total_loss / len(train_loader)

    # 每 10 个 epoch 输出一次训练进度
    if (epoch + 1) % 10 == 0:
        print(f'epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

# ==================== 训练完成 ====================
print("训练完成！")

# ==================== 可选：模型评估 ====================
# 切换到评估模式
model.eval()
with torch.no_grad():
    # 在测试集上预测
    test_outputs = model(X_test_tensor)
    test_outputs = (test_outputs > 0.5).float()  # 阈值 0.5 得到预测类别
    accuracy = (test_outputs.view(-1) == Y_test_tensor).float().mean()
    print(f"测试集准确率: {accuracy:.4f}")