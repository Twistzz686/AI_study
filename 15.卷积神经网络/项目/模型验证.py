import os
import numpy as np

import random

import torch
from torchvision import datasets,transforms
from main import LeNet5


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available,Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available,Using CPU.")

train_dataset = datasets.MNIST(root="./dataset",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root="./dataset",train=False,transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)

model = LeNet5().to(device)
cri = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

# epoches = 10
# for epoch in range(epoches):
#     model.train()
#     total_loss = 0
#     for i,(images,labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = model(images)
#         loss = cri(outputs,labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch + 1}/{epoches},Loss:{avg_loss:.4f}")

model.load_state_dict(torch.load("model.pth"))

total = 0
correct = 0

with torch.no_grad():
    for images,labels in test_loader:
        images.to(device)
        labels.to(device)
        outputs = model(images)

        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the model on the test images {100 * correct / total}%")