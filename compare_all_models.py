"""
compare_all_models.py - 四模型压缩性能完整对比
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import *

print("="*60)
print("四模型压缩性能完整对比")
print("="*60)

# ==================== 模型路径 ====================
MODELS = {
    "Baseline": os.path.join(MODEL_DIR, "xception_baseline.pth"),
    "LS-only": os.path.join(MODEL_DIR, "xception_ls_only.pth"),
    "JPEG-only": os.path.join(MODEL_DIR, "xception_jpeg_only.pth"),
    "JPEG+LS": os.path.join(MODEL_DIR, "xception_ls_jpeg.pth")
}

qualities = [100, 90, 70, 50, 30]

# ==================== 评估函数 ====================
def evaluate_model(model_path, val_path):
    model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
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

# ==================== 评估所有模型 ====================
all_results = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️ 模型不存在: {name}")
        continue
    
    print(f"\n评估 {name}...")
    results = []
    
    for qf in qualities:
        val_path = os.path.join(COMPRESSED_DIR, f"qf_{qf}")
        if not os.path.exists(val_path):
            results.append(None)
            continue
        acc = evaluate_model(path, val_path)
        results.append(round(acc * 100, 2))
        print(f"  QF={qf:3d}: {acc:.2%}")
    
    all_results[name] = results

# ==================== 保存结果 ====================
df = pd.DataFrame(all_results, index=qualities)
df.index.name = "QF"
df.to_csv(os.path.join(RESULT_DIR, "all_models_comparison.csv"))
print(f"\n✅ 数据已保存: {RESULT_DIR}/all_models_comparison.csv")

# ==================== 绘制对比图 ====================
plt.figure(figsize=(12, 7))

colors = {
    "Baseline": "#2E86AB",
    "LS-only": "#E66A2C", 
    "JPEG-only": "#28A745",
    "JPEG+LS": "#C5283D"
}
markers = {
    "Baseline": "o",
    "LS-only": "s",
    "JPEG-only": "^",
    "JPEG+LS": "D"
}
linestyles = {
    "Baseline": "-",
    "LS-only": "--",
    "JPEG-only": "-.",
    "JPEG+LS": "-"
}

for name, results in all_results.items():
    if results and None not in results:
        plt.plot(qualities, results,
                 marker=markers.get(name, "o"),
                 linewidth=2.5,
                 markersize=8,
                 color=colors.get(name, "#333"),
                 linestyle=linestyles.get(name, "-"),
                 label=name)

plt.xlabel('压缩等级 (QF值)', fontsize=12)
plt.ylabel('检测准确率 (%)', fontsize=12)
plt.title('四模型压缩性能对比', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.ylim(55, 100)

# 保存图片
save_path = os.path.join(RESULT_DIR, "all_models_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存: {save_path}")

plt.show()