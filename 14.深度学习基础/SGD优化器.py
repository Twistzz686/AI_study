import numpy as np
from IPython import display

# ==================== 1. 数据准备 ====================
# 数据集：每行为 [x, y] 坐标对，y ≈ 50x + 30 左右的线性关系
points = np.array([[-0.5, 7.7],
                   [1.8, 98.5],
                   [0.9, 57.8],
                   [0.4, 39.2],
                   [-1.4, -15.7],
                   [-1.4, -37.3],
                   [-1.8, -49.1],
                   [1.5, 75.6],
                   [0.4, 34.0],
                   [0.8, 62.3]])

# 提取特征 X（第一列）和标签 Y（第二列）
X = points[:, 0]  # 特征值，形状 (10,)
Y = points[:, 1]  # 标签值，形状 (10,)

# ==================== 2. 初始化参数 ====================
w = 0  # 权重初始值
b = -1  # 偏置初始值
lr = 0.01  # 学习率


# ==================== 3. 损失函数 ====================
def loss_func(X, w, b):
    """
    计算均方误差损失
    X: 特征值数组
    w: 权重
    b: 偏置

    ⚠️ 问题：函数内部使用了全局变量 Y，应该作为参数传入
    """
    pre_y = np.dot(X, w) + b  # 预测值 = w*x + b
    total_loss = np.mean((pre_y - Y) ** 2)  # ❌ Y 是全局变量，应改为参数
    return total_loss


# ==================== 4. 随机梯度下降函数 ====================
def SGD(points, w, b, lr, batch_size):
    """
    执行一个 epoch 的随机梯度下降
    points: 数据集（包含 X 和 Y）
    w, b: 模型参数
    lr: 学习率
    batch_size: 批次大小
    """
    # 打乱数据顺序，增加随机性
    np.random.shuffle(points)

    # 分批处理
    for num_batch in range(0, len(points), batch_size):
        # 获取当前批次的数据
        batch_points = points[num_batch:num_batch + batch_size, :]
        batch_x = batch_points[:, 0]  # 批次特征
        batch_y = batch_points[:, 1]  # 批次标签

        # 前向传播：计算预测值
        batch_pre_y = w * batch_x + b

        # ========== 梯度计算 ==========
        # ✅ 修正后的梯度公式（已正确）
        # 损失函数 MSE: L = (1/n) * Σ(pred - y)²
        # ∂L/∂w = (2/n) * Σ(pred - y) * x
        # ∂L/∂b = (2/n) * Σ(pred - y)
        dw = np.mean(2 * (batch_pre_y - batch_y) * batch_x)
        db = np.mean(2 * (batch_pre_y - batch_y))

        # 参数更新（梯度下降）
        w -= lr * dw
        b -= lr * db

    return w, b


# ==================== 5. 训练循环 ====================
epoches = 1000  # 训练轮数
bs = 10  # 批次大小（等于数据集大小，实际是批量梯度下降）

for epoch in range(1, epoches + 1):
    # ⚠️ 问题：没有接收 SGD 的返回值！
    # SGD 返回更新后的 w, b，但没有赋值给当前变量
    SGD(points, w, b, lr, batch_size=bs)  # ❌ 返回值被丢弃

    # 每 20 个 epoch 或第 1 个 epoch 打印损失
    if epoch == 1 or epoch % 20 == 0:
        print(loss_func(X, w, b))  # w, b 从未被更新，始终是初始值！

# ==================== 代码问题总结 ====================
#
# 🔴 问题1：SGD 返回值被丢弃
#    正确写法：w, b = SGD(points, w, b, lr, batch_size=bs)
#
# 🔴 问题2：loss_func 依赖全局变量 Y
#    正确写法：def loss_func(X, Y, w, b): ...
#
# 🟡 问题3：batch_size=10 等于数据集大小，实际是批量梯度下降而非 SGD
#    真正的 SGD 应该 batch_size=1
#
# ==================== 修正后的代码 ====================
"""
def loss_func(X, Y, w, b):
    pre_y = np.dot(X, w) + b
    return np.mean((pre_y - Y) ** 2)

for epoch in range(1, epoches + 1):
    w, b = SGD(points, w, b, lr, batch_size=bs)  # ✅ 接收返回值
    if epoch == 1 or epoch % 20 == 0:
        print(loss_func(X, Y, w, b))  # ✅ 传入 Y 参数
"""