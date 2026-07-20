"""
confidence_analysis.py - 分析模型预测置信度分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import os
from config import *

print("="*60)
print("模型置信度分析")
print("="*60)

# ==================== 配置 ====================
MODELS = {
    "Baseline": os.path.join(MODEL_DIR, "xception_baseline.pth"),
    "LS-only": os.path.join(MODEL_DIR, "xception_ls_only.pth"),
}

# 使用的压缩等级
COMPRESS_QUALITY = 30

# 选择验证集
val_path = os.path.join(COMPRESSED_DIR, f"qf_{COMPRESS_QUALITY}")
print(f"验证集: {val_path}")

# ==================== 提取置信度 ====================
def get_confidences(model_path, val_path):
    """获取模型在验证集上的预测置信度"""
    model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    dataset = datasets.ImageFolder(val_path, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_confidences = []
    all_correct = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # 最大概率作为置信度
            confidences = probs.max(dim=1)[0]
            correct = (preds == labels)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        "confidences": np.array(all_confidences),
        "correct": np.array(all_correct),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels)
    }

all_results = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️ 模型不存在: {name}")
        continue
    
    print(f"\n提取 {name} 置信度...")
    results = get_confidences(path, val_path)
    all_results[name] = results
    
    print(f"  平均置信度: {results['confidences'].mean():.2%}")
    print(f"  准确率: {results['correct'].mean():.2%}")

# ==================== 绘制置信度密度分布 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, results) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # 正确预测的置信度分布
    correct_conf = results['confidences'][results['correct']]
    # 错误预测的置信度分布
    wrong_conf = results['confidences'][~results['correct']]
    
    # 核密度估计
    x = np.linspace(0.5, 1.0, 100)
    
    if len(correct_conf) > 0:
        kde_correct = gaussian_kde(correct_conf)
        ax.plot(x, kde_correct(x), label='正确预测', color='#2E86AB', linewidth=2.5)
    
    if len(wrong_conf) > 0:
        kde_wrong = gaussian_kde(wrong_conf)
        ax.plot(x, kde_wrong(x), label='错误预测', color='#C5283D', linewidth=2.5)
    
    ax.set_xlabel('预测置信度', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title(f'{name} 置信度分布 (QF={COMPRESS_QUALITY})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.0)

plt.suptitle('基线 vs LS-only 置信度分布对比', fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(RESULT_DIR, f"confidence_compare_qf{COMPRESS_QUALITY}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存: {save_path}")

plt.show()

# ==================== 打印关键指标 ====================
print("\n" + "="*60)
print("置信度关键指标")
print("="*60)

for name, results in all_results.items():
    correct_conf = results['confidences'][results['correct']]
    wrong_conf = results['confidences'][~results['correct']]
    
    print(f"\n{name}:")
    print(f"  正确预测平均置信度: {correct_conf.mean():.2%}" if len(correct_conf) > 0 else "  无正确预测")
    print(f"  错误预测平均置信度: {wrong_conf.mean():.2%}" if len(wrong_conf) > 0 else "  无错误预测")
    
    if len(correct_conf) > 0 and len(wrong_conf) > 0:
        print(f"  置信度差距: {(correct_conf.mean() - wrong_conf.mean()):.2%}")