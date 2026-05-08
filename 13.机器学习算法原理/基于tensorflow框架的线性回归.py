import tensorflow as tf
import numpy as np



data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]

data = np.array(data)
x_data = data[:,0]
y_data = data[:,1]

x_train = tf.constant(x_data,dtype=tf.float32)
y_train = tf.constant(y_data,dtype=tf.float32)

# (1,)指的是(None,1)形状
model = tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=(1,))])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,loss="mean_squared_error")

epoches = 500
history = model.fit(x_train,y_train,epochs=epoches)
# for epoch in range(1,epoches + 1):
#         history = model.fit(x_train,y_train)
#         loss = history.history["loss"][0]
#
#         if epoch % 10 == 0 or epoch == 1:
#                 print(f"epoch:{epoch},loss:{loss}")