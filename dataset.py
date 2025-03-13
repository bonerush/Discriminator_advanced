import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms)

train_loader = DataLoader(train_dataset, 
                          batch_size=64, 
                          shuffle=True)
# shuffle=True表示每个epoch都打乱数据集

test_loader = DataLoader(test_dataset,  
                         batch_size=64, 
                         shuffle=True)  

