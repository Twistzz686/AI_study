import numpy as np
from IPython import display
import matplotlib.pyplot as plt

# ==================== 1. 数据准备 ====================
# 第一类数据点（标签为0）
class1_points = np.array([[1.9, 1.2],
                          [1.5, 2.1],
                          [1.9, 0.5],
                          [1.5, 0.9],
                          [0.9, 1.2],
                          [1.1, 1.7],
                          [1.4, 1.1]])

# 第二类数据点（标签为1）
class2_points = np.array([[3.2, 3.2],
                          [3.7, 2.9],
                          [3.2, 2.6],
                          [1.7, 3.3],
                          [3.4, 2.6],
                          [4.1, 2.3],
                          [3.0, 2.9]])

# 提取所有样本的第一个特征（x坐标）
x1_data = np.concatenate([class1_points[:, 0], class2_points[:, 0]])

# 提取所有样本的第二个特征（y坐标）
x2_data = np.concatenate([class1_points[:, 1], class2_points[:, 1]])

# 生成标签：前7个为0（第一类），后7个为1（第二类）
label = np.concatenate([np.zeros(len(class1_points)), np.ones(len(class2_points))])


# ==================== 2. 定义 Sigmoid 函数 ====================
def sigmoid(x):
    """Sigmoid激活函数，将任意实数映射到(0,1)区间"""
    return 1 / (1 + np.exp(-x))


# ==================== 3. 前向传播函数 ====================
def forward(w1, w2, b):
    """
    前向传播计算预测概率
    w1: 第一个特征的权重
    w2: 第二个特征的权重
    b: 偏置
    返回: 预测概率 a = sigmoid(w1*x1 + w2*x2 + b)
    """
    z = w1 * x1_data + w2 * x2_data + b  # 线性组合
    a = sigmoid(z)  # 通过Sigmoid得到概率
    return a


# ==================== 4. 初始化参数 ====================
w1 = 0.1  # 特征1的权重初始值
w2 = 0.1  # 特征2的权重初始值
b = 0  # 偏置初始值
lr = 0.05  # 学习率


# ==================== 5. 定义损失函数（交叉熵） ====================
def loss_function(a):
    """
    计算二分类交叉熵损失
    a: 预测概率（Sigmoid输出）
    返回: 平均交叉熵损失
    """
    # 防止log(0)出现无穷大，加微小值1e-8
    loss = -np.mean(label * np.log(a + 1e-8) + (1 - label) * np.log(1 - a + 1e-8))
    return loss


# ==================== 6. 训练循环 ====================
epoches = 1000  # 训练轮数

for epoch in range(1, epoches + 1):
    # ----- 前向传播 -----
    a = forward(w1, w2, b)  # 预测概率

    # ----- 反向传播（链式法则）-----
    # 损失对激活值的导数 ∂L/∂a = (a - y) / (a(1-a))
    deda = (a - label) / (a * (1 - a))

    # Sigmoid的导数 ∂a/∂z = a(1-a)
    dadz = a * (1 - a)

    # 线性部分对各参数的导数
    dzdw1 = x1_data  # ∂z/∂w1 = x1
    dzdw2 = x2_data  # ∂z/∂w2 = x2
    dzdb = 1  # ∂z/∂b = 1

    # 计算各参数的梯度（注意：deda * dadz 化简后 = a - label）
    # 使用 np.dot 实现元素相乘再求和，然后除以样本数得到平均梯度
    gradient_w1 = np.dot(dzdw1, (deda * dadz)) / len(x1_data)
    gradient_w2 = np.dot(dzdw2, (deda * dadz)) / len(x2_data)
    gradient_b = (deda * dadz * dzdb).sum() / len(x1_data)

    # ----- 参数更新（梯度下降）-----
    w1 -= lr * gradient_w1
    w2 -= lr * gradient_w2
    b -= lr * gradient_b

    # ----- 输出训练进度 -----
    if epoch % 50 == 0 or epoch == 1:
        a = forward(w1, w2, b)  # 用更新后的参数重新计算预测值
        loss = loss_function(a)  # 计算当前损失
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# ==================== 训练完成 ====================
print(f"\n训练完成！最终参数：w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")