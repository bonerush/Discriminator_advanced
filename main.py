import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

# 固定所有随机种子以确保结果可重复
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True  # 保证卷积结果一致（可能降低性能）
torch.backends.cudnn.benchmark = False     # 关闭cuDNN的自动优化

from dataset import train_loader, test_loader
from discriminator import Discriminator

BATCH_SIZE = 128
IN_PLANES = 28*28
N_LAYERS = 2
HIDDEN = 512
DIRECT_NUM = 10
EPOCHS = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Discriminator(IN_PLANES, N_LAYERS, HIDDEN, DIRECT_NUM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 使用tqdm添加进度条
    print(f"---------- Epoch {epoch+1}/{EPOCHS} ----------")
    train_iterator = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        dynamic_ncols=True
    )
    
    for images, labels in train_iterator:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # 更新进度条描述
        train_iterator.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * total_correct / total_samples:.2f}%"
        })
    
    # 计算epoch统计信息
    avg_loss = total_loss / total_samples
    epoch_acc = 100 * total_correct / total_samples
    # 打印epoch结果
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Avg Loss: {avg_loss:.4f} | "
          f"Train Acc: {epoch_acc:.2f}%\n")

# 最终测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")