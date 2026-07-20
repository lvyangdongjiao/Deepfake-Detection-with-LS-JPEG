"""
visualize_tsne_compare.py - 多个模型t-SNE对比
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from config import *
from extract_features import get_model_features

# ==================== 配置 ====================
# 要对比的模型
MODELS = {
    "Baseline": os.path.join(MODEL_DIR, "xception_baseline.pth"),
    "LS-only": os.path.join(MODEL_DIR, "xception_ls_only.pth"),
    "JPEG-only": os.path.join(MODEL_DIR, "xception_jpeg_only.pth"),
    "JPEG+LS": os.path.join(MODEL_DIR, "xception_ls_jpeg.pth")
}

# 使用的压缩等级
COMPRESS_QUALITY = 30
SAMPLE_SIZE = 200

# ==================== 提取所有模型特征 ====================
print("="*60)
print("多模型t-SNE对比 (QF=30)")
print("="*60)

val_path = os.path.join(COMPRESSED_DIR, f"qf_{COMPRESS_QUALITY}")

if not os.path.exists(val_path):
    print(f"❌ 压缩验证集不存在: {val_path}")
    exit()

all_features = {}
all_labels = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️ 模型不存在: {name}")
        continue
    
    print(f"\n提取 {name} 特征...")
    features, labels = get_model_features(path, val_path, device, BATCH_SIZE)
    
    if features is None:
        continue
    
    # 随机采样
    if len(features) > SAMPLE_SIZE:
        np.random.seed(42)
        indices = np.random.choice(len(features), SAMPLE_SIZE, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    all_features[name] = features
    all_labels[name] = labels

# ==================== t-SNE降维 ====================
print("\n运行t-SNE...")
tsne_results = {}

for name, features in all_features.items():
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results[name] = tsne.fit_transform(features)
    print(f"  {name}: 完成")

# ==================== 绘制对比图 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

colors = ['#2E86AB', '#C5283D']
labels_name = ['真实', '伪造']

for idx, (name, features_2d) in enumerate(tsne_results.items()):
    ax = axes[idx]
    labels = all_labels[name]
    
    for i in range(2):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=labels_name[i],
            alpha=0.6,
            s=25
        )
    
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

plt.suptitle(f'不同模型在QF={COMPRESS_QUALITY}下的t-SNE特征分布对比', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(RESULT_DIR, f"tsne_compare_qf{COMPRESS_QUALITY}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存: {save_path}")

plt.show()