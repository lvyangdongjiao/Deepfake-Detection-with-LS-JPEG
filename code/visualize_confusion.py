"""
visualize_confusion.py - 混淆矩阵可视化
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import *

# ==================== 配置 ====================
MODEL_NAME = "Baseline"
MODEL_PATH = os.path.join(MODEL_DIR, "xception_baseline.pth")

# 使用压缩等级 (None表示原始验证集)
COMPRESS_QUALITY = 30

# ==================== 模型预测 ====================
print("="*60)
print(f"混淆矩阵: {MODEL_NAME}")
print("="*60)

# 选择验证集
if COMPRESS_QUALITY:
    val_path = os.path.join(COMPRESSED_DIR, f"qf_{COMPRESS_QUALITY}")
    title_suffix = f"(QF={COMPRESS_QUALITY})"
else:
    val_path = VAL_PATH
    title_suffix = "(原始)"

print(f"验证集: {val_path}")

# 加载模型
model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# 预测
dataset = datasets.ImageFolder(val_path, transform=val_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==================== 计算混淆矩阵 ====================
cm = confusion_matrix(all_labels, all_preds)

print(f"\n混淆矩阵:")
print(f"              预测")
print(f"             真实  伪造")
print(f"真实    {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"伪造    {cm[1,0]:6d}  {cm[1,1]:6d}")

# 计算指标
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n评估指标:")
print(f"  准确率: {accuracy:.2%}")
print(f"  精确率: {precision:.2%}")
print(f"  召回率: {recall:.2%}")
print(f"  F1分数: {f1:.2%}")

# 假阳性率：真实被误判为伪造的比例
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
# 假阴性率：伪造被误判为真实的比例
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"  假阳性率: {fpr:.2%} (真实→伪造)")
print(f"  假阴性率: {fnr:.2%} (伪造→真实)")

# ==================== 绘制混淆矩阵 ====================
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['真实', '伪造'],
    yticklabels=['真实', '伪造'],
    cbar=True,
    square=True,
    annot_kws={'fontsize': 16}
)

plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.title(f'{MODEL_NAME} 混淆矩阵 {title_suffix}', fontsize=14)

# 添加指标文本
metrics_text = f"准确率: {accuracy:.2%}  |  精确率: {precision:.2%}  |  召回率: {recall:.2%}"
plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10)

plt.tight_layout()

save_path = os.path.join(RESULT_DIR, f"confusion_{MODEL_NAME}_qf{COMPRESS_QUALITY}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存: {save_path}")

plt.show()
