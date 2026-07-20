"""
visualize_tsne.py - t-SNE特征可视化
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from config import *
from extract_features import get_model_features

# ==================== 配置 ====================
# 要可视化的模型
MODEL_NAME = "Baseline"
MODEL_PATH = os.path.join(MODEL_DIR, "xception_baseline.pth")

# 使用的压缩等级 (None表示原始验证集)
COMPRESS_QUALITY = 30  # 使用QF=30的压缩验证集

# 从验证集中随机抽取的样本数（提高可视化速度）
SAMPLE_SIZE = 200

# ==================== 加载特征 ====================
print("="*60)
print(f"t-SNE可视化: {MODEL_NAME}")
print("="*60)

# 选择验证集路径
if COMPRESS_QUALITY:
    val_path = os.path.join(COMPRESSED_DIR, f"qf_{COMPRESS_QUALITY}")
    title_suffix = f"(QF={COMPRESS_QUALITY})"
else:
    val_path = VAL_PATH
    title_suffix = "(原始)"

print(f"验证集: {val_path}")

# 提取特征
features, labels = get_model_features(MODEL_PATH, val_path, device, BATCH_SIZE)

if features is None:
    print("❌ 特征提取失败")
    exit()

# 随机采样（如果样本太多）
if len(features) > SAMPLE_SIZE:
    np.random.seed(42)
    indices = np.random.choice(len(features), SAMPLE_SIZE, replace=False)
    features = features[indices]
    labels = labels[indices]
    print(f"随机采样 {SAMPLE_SIZE} 个样本")

# ==================== 运行t-SNE ====================
print("运行t-SNE降维...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42,
    n_iter=1000
)
features_2d = tsne.fit_transform(features)
print(f"t-SNE完成，特征已降至2维")

# ==================== 绘制 ====================
plt.figure(figsize=(10, 8))

# 真实样本 (label=0) 用蓝色，伪造样本 (label=1) 用红色
colors = ['#2E86AB', '#C5283D']
labels_name = ['真实', '伪造']

for i in range(2):
    mask = labels == i
    plt.scatter(
        features_2d[mask, 0], 
        features_2d[mask, 1],
        c=colors[i],
        label=labels_name[i],
        alpha=0.6,
        s=30
    )

plt.xlabel('t-SNE 维度 1', fontsize=12)
plt.ylabel('t-SNE 维度 2', fontsize=12)
plt.title(f'{MODEL_NAME} t-SNE特征可视化 {title_suffix}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

# 保存图片
save_path = os.path.join(RESULT_DIR, f"tsne_{MODEL_NAME}_qf{COMPRESS_QUALITY}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图片已保存: {save_path}")

plt.show()