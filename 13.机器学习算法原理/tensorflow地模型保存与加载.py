import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import models



data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]


data = np.array(data)
x_data = data[:,0]
y_data = data[:,1]

x_train = tf.constant(np.expand_dims(x_data,axis=1),dtype=tf.float32)
y_train = tf.constant(y_data,dtype=tf.float32)

# 方法1
# (1,)指的是(None,1)形状
# model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,))])


# 方法2
# model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=(1,)))
# model.add(tf.keras.layers.Dense(1))

# 方案3
# class Linear(Model):
#     def __init__(self):
#         super(Linear,self).__init__()
#         self.linear = tf.keras.layers.Dense(1)
#     def __call__(self,x,**kwargs):
#         x = self.linear(x)
#         return x
#
# model = Linear

# 方案4
def linear():
    input = tf.keras.layers.Input(shape=(1,),dtype=tf.float32)
    y = tf.keras.layers.Dense(1)(input)
    model = tf.keras.models.Model(input=input,output=y)
    return model
model = linear()


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,loss="mean_squared_error")

epoches = 500
history = model.fit(x_train,y_train,epochs=epoches)


# 1 HDF5文件,模型必须是函数模型或顺序模型
model.save("./my_model.h5")

loaded_model = models.load_model("my_model.h5")

# 2  只保存参数
model.save_weights("model_weights.h5")

# 3  savemodel格式，tensorflow序列化格式，适合在不同的tensorflow环境中部署
model.save("./saved_model",save_format="tf")