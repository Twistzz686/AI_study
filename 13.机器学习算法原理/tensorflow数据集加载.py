import tensorflow as tf
import numpy as np

# ==================== 1. 数据准备 ====================
# 原始数据：每个子列表为 [x, y] 坐标对
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]

# 转换为 NumPy 数组，便于数据处理
data = np.array(data)

# 提取特征 x（第一列）和标签 y（第二列）
x_data = data[:, 0]  # x: [-0.5, 1.8, 0.9, 0.4, -1.4, -1.4, -1.8, 1.5, 0.4, 0.8]
y_data = data[:, 1]  # y: [7.7, 98.5, 57.8, 39.2, -15.7, -37.3, -49.1, 75.6, 34.0, 62.3]

# ==================== 2. 转换为 TensorFlow 张量 ====================
# 将 NumPy 数组转换为 TensorFlow 常量张量，数据类型为 float32
x_train = tf.constant(x_data, dtype=tf.float32)
y_train = tf.constant(y_data, dtype=tf.float32)

# ==================== 3. 创建数据集 ====================
# 从张量切片创建数据集，将 x 和 y 配对
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# 打乱数据顺序：buffer_size=10 表示从10个样本中随机采样
# 防止模型学习到数据的原始顺序
dataset = dataset.shuffle(buffer_size=10)

# 批处理：每批10个样本（因为总共10个样本，所以只有1批）
dataset = dataset.batch(10)

# 预取数据：在GPU训练的同时，CPU预先加载下一批数据
# AUTOTUNE 让 TensorFlow 自动选择最优的预取数量
# 注意：在纯CPU训练时效果不明显
# dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==================== 4. 创建模型 ====================
# Sequential: 顺序模型，层按顺序堆叠
# Dense(1): 全连接层，输出1个神经元，实现 y = w*x + b
# input_shape=(1,): 输入形状是1维特征（每个样本只有1个x值）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# ==================== 5. 编译模型 ====================
# SGD: 随机梯度下降优化器，学习率=0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# compile: 配置模型训练
# optimizer: 优化器，用于更新参数
# loss: 损失函数，mean_squared_error 是均方误差（MSE）
model.compile(optimizer=optimizer, loss="mean_squared_error")

# ==================== 6. 训练模型 ====================
epoches = 500  # 训练轮数（整个数据集被完整遍历的次数）

# fit: 开始训练
# dataset: 训练数据集（注意：这里应该传 dataset，不是 x_train, y_train）
# epochs: 训练轮数
history = model.fit(dataset, epochs=epoches)

# ==================== 7. 训练结果说明 ====================
# 训练完成后，可以查看学习到的参数：
# weights = model.layers[0].get_weights()
# print(f"权重 w: {weights[0][0][0]:.4f}")
# print(f"偏置 b: {weights[1][0]:.4f}")

# history 对象记录了每个 epoch 的损失值
# print(history.history['loss'])  # 打印所有 epoch 的损失


