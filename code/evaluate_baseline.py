"""
evaluate_baseline.py - 评估基线模型在不同压缩等级下的性能
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import pandas as pd
from config import *

print("="*60)
print("基线模型压缩性能评估")
print("="*60)

# ==================== 加载模型 ====================
model_path = os.path.join(MODEL_DIR, "xception_baseline.pth")
if not os.path.exists(model_path):
    print(f"❌ 模型不存在: {model_path}")
    print("请先运行 train_baseline.py")
    exit()

model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

print(f"✅ 模型加载成功: {model_path}")

# ==================== 评估函数 ====================
def evaluate(val_path):
    dataset = datasets.ImageFolder(val_path, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ==================== 评估所有压缩等级 ====================
qualities = [100, 90, 70, 50, 30]
results = []

print("\n评估不同压缩等级:")
print("-" * 40)

for qf in qualities:
    val_path = os.path.join(COMPRESSED_DIR, f"qf_{qf}")
    
    if not os.path.exists(val_path):
        print(f"  QF={qf}: 压缩验证集不存在，请先运行 generate_compressed_val.py")
        continue
    
    acc = evaluate(val_path)
    results.append({"QF": qf, "准确率": round(acc * 100, 2)})
    print(f"  QF={qf:3d}: {acc:.2%}")

# ==================== 保存结果 ====================
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT_DIR, "baseline_compression_results.csv"), index=False)
print(f"\n✅ 结果已保存: {RESULT_DIR}/baseline_compression_results.csv")

# ==================== 打印关键发现 ====================
print("\n" + "="*60)
print("关键发现")
print("="*60)

if len(results) >= 2:
    high_acc = results[0]["准确率"]  # QF=100
    low_acc = results[-1]["准确率"]   # QF=30
    drop = high_acc - low_acc
    print(f"📉 性能下降: {high_acc:.2f}% → {low_acc:.2f}% (下降 {drop:.2f}%)")
    
    if drop > 30:
        print("⚠️ 性能下降超过30%，模型在压缩场景下严重退化")
