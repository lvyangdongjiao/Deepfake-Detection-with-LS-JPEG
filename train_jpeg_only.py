"""
train_jpeg_only.py - 训练JPEG-only模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import pandas as pd
import os
from config import *
from jpeg_dataloader import get_jpeg_train_loader

print("="*60)
print("开始训练 JPEG-only 模型")
print("="*60)

# ==================== 参数 ====================
JPEG_QUALITY_RANGE = (30, 90)  # 压缩质量范围
JPEG_PROB = 0.5                # 压缩概率

print(f"JPEG增强配置:")
print(f"  质量范围: {JPEG_QUALITY_RANGE[0]} - {JPEG_QUALITY_RANGE[1]}")
print(f"  压缩概率: {JPEG_PROB}")

# ==================== 加载数据 ====================
train_loader = get_jpeg_train_loader(
    quality_range=JPEG_QUALITY_RANGE,
    prob=JPEG_PROB
)

# 验证集（无增强）
val_dataset = datasets.ImageFolder(VAL_PATH, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集大小: {len(train_loader.dataset)} 张")
print(f"验证集大小: {len(val_loader.dataset)} 张")

# ==================== 创建模型 ====================
model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# ==================== 评估函数 ====================
def evaluate():
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

print("\n开始训练...")
print("-" * 60)

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
    
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "xception_jpeg_only.pth"))
        print(f"  ✅ 保存最佳模型 (Val: {acc:.2%})")

print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2%}")

# ==================== 保存训练记录 ====================
df = pd.DataFrame(history)
df.to_csv(os.path.join(MODEL_DIR, "jpeg_only_history.csv"), index=False)
print(f"训练记录已保存: {MODEL_DIR}/jpeg_only_history.csv")