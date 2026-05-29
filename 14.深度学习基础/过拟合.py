import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
import torch.optim as optim

points = np.array([[-0.5, 7.7], [1.2, 65.8], [0.4, 39.2], [-1.4, -15.7], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]])

x_train = points[:,0]
y_train = points[:,1]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.layer1 = nn.Linear(1,16)
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = (self.layer4(x))
        return x

model = Model()
lr = 0.05
cri = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)


epoches = 1000
for epoch in range(1,epoches + 1):
    x_train_tensor = torch.tensor(x_train,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train,dtype=torch.float32)

    y_pred = model(x_train_tensor.unsqueeze(1))

    loss = cri(y_pred.squeeze(1),y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 20 == 0:
        print(f"epoch:{epoch},loss:{loss}")