import numpy as np
from IPython import display
import matplotlib.pyplot as plt

# ==================== 1. 数据准备 ====================
# 原始数据：[x值, 标签]（0表示负类，1表示正类）
# 这是一个简单的二分类问题，数据分布在x≈0.8-1.7时为0，x≈3.2-4.2时为1
data = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [3.2, 1], [3.7, 1], [4.0, 1], [4.2, 1]])

# 提取特征 x 和标签 y
x_data = data[:, 0]  # 特征数据
y_data = data[:, 1]  # 标签数据（0或1）

# 转换为 numpy 数组
x_train = np.array(x_data)
y_train = np.array(y_data)


# ==================== 2. 定义 Sigmoid 函数 ====================
def sigmoid(x):
    """
    注意：标准的 Sigmoid 函数是 1/(1+exp(-x))
    这里的实现是 x/(1+exp(-x))，这是错误的！会导致输出范围不是(0,1)
    正确应为：return 1 / (1 + np.exp(-x))
    """
    return 1 / (1 + np.exp(-x))  


# ==================== 3. 初始化参数 ====================
w = 0  # 权重初始化为0
b = 0  # 偏置初始化为0
learning_rate = 0.1  # 学习率

# ==================== 4. 可视化准备 ====================
fig, (ax1, ax2) = plt.subplots(2, 1)  # 创建2个子图：上面显示数据+决策边界，下面显示损失曲线
epoch_list = []  # 记录epoch
loss_list = []  # 记录损失值

# ==================== 5. 训练循环 ====================
epoches = 1000  # 训练轮数

for epoch in range(1, epoches + 1):
    # ----- 前向传播 -----
    # 计算线性组合 z = w*x + b
    z = w * x_train + b

    # 通过 Sigmoid 激活函数得到预测概率 a = sigmoid(z)
    a = sigmoid(z)

    # ----- 反向传播（手动计算梯度）-----
    # 损失函数：MSE = (y_true - a)^2 的平均值
    # da/d? 推导过程：
    # loss = (y - a)^2
    # dloss/da = -2 * (y - a)
    deda = -2 * (y_train - a)  # 损失对激活值的导数

    # Sigmoid 函数的导数：da/dz = a * (1 - a)
    dadz = a * (1 - a)

    # dz/dw = x
    dzdw = x_train
    # 链式法则：损失对权重的梯度 = mean(deda * dadz * dzdw)
    gradient_w = np.mean(deda * dadz * dzdw)

    # dz/db = 1
    dzdb = 1
    # 损失对偏置的梯度
    gradient_b = np.mean(deda * dadz * dzdb)

    # ----- 参数更新（梯度下降）-----
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    # ----- 计算当前损失 -----
    z = w * x_train + b
    a = sigmoid(z)
    loss = np.mean((y_train - a) ** 2)  # 均方误差损失

    # 记录损失历史
    epoch_list.append(epoch)
    loss_list.append(loss)

    # ----- 输出训练状态 -----
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch:{epoch}, loss:{loss:.6f}")

        # 生成用于绘图的 x 值范围
        x_min = x_data.min()
        x_max = x_data.max()
        x_values = np.linspace(x_min, x_max, int((x_max - x_min) * 10))

        # 计算决策边界的 y 值（预测概率）
        y_values = np.round(sigmoid(w * x_values + b), 3)

        # ----- 更新子图1：散点图 + 决策曲线 -----
        ax1.clear()
        ax1.scatter(x_data, y_data, color='blue', label='data points')
        ax1.plot(x_values, y_values, color='red', label='decision boundary')
        ax1.set_xlabel('x')
        ax1.set_ylabel('prediction')
        ax1.set_title('Logistic Regression')
        ax1.legend()

        # ----- 更新子图2：损失曲线 -----
        ax2.clear()
        ax2.plot(epoch_list, loss_list, color='green')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
        ax2.set_title('Loss vs Epoch')

        # 暂停以更新图形
        plt.pause(1)

# 显示最终图形
plt.show()