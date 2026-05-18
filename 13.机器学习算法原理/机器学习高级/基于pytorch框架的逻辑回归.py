import numpy as np
import torch
from torch import nn

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

x_train = np.concatenate((class1_points,class2_points))
y_train = np.concatenate((np.zeros(len(class1_points)),np.ones(len(class2_points))))

torch.manual_seed(1)


class LogisticRegreModel(nn.Module):
    def __init__(self):
        super(LogisticRegreModel,self).__init__()
        self.fc = torch.nn.Linear(2,1)


    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)

model = LogisticRegreModel()

cri = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)


epoches = 1000
for epoch in range(1,epoches + 1):
    inputs = torch.tensor(x_train,dtype=torch.float32)
    labels = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)

    outputs = model(inputs)
    loss = cri(outputs,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch:{epoch},loss:{loss}")