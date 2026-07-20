"""
train_baseline.py - 训练Xception基线模型
"""

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import pandas as pd
from config import *

print("="*60)
print("开始训练基线模型 (Xception)")
print("="*60)

# ==================== 加载数据 ====================
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"训练集大小: {len(train_dataset)} 张")
print(f"类别: {train_dataset.classes}")

# ==================== 创建模型 ====================
model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==================== 评估函数 ====================
def evaluate():
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ==================== 训练循环 ====================
best_acc = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    acc = evaluate()
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val: {acc:.2%}")
    
    history.append({"epoch": epoch+1, "loss": avg_loss, "val_acc": acc})
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "xception_baseline.pth"))
        print(f"  ✅ 保存最佳模型 (Val: {acc:.2%})")

print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2%}")

# ==================== 保存训练记录 ====================
df = pd.DataFrame(history)
df.to_csv(os.path.join(MODEL_DIR, "baseline_history.csv"), index=False)
print(f"训练记录已保存: {MODEL_DIR}/baseline_history.csv")