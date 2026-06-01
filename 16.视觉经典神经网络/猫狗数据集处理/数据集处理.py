import os
import random
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets,transforms

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU")
else:
    device = torch.device("cpu")
    print("CUDA is NOT available. Using CPU")

transform = {
    "train":transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "test":transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
}
train_dataset = datasets.ImageFolder("./dataset/train",transform=transform["train"])
test_dataset = datasets.ImageFolder("./dataset/test",transform=transform["test"])

train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=True)

examples = enumerate(test_dataloader)
batch_idx,(imgs,labels) = next(examples)

for i in range(4):
    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])
    image = imgs[i].numpy() * std[:,None,None] + mean[:,None,None]
    image = np.transpose(image,(1,2,0))
    plt.subplot(2,2,i+1)
    plt.imshow(image)
    plt.title(f"Truth:{labels[i]}")
plt.show()