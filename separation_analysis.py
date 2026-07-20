"""
separation_analysis.py - 特征分离度与5-NN准确率量化分析
"""

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from config import *
from extract_features import get_model_features

print("="*60)
print("特征分离度与5-NN准确率分析")
print("="*60)

# ==================== 配置 ====================
MODELS = {
    "Baseline": os.path.join(MODEL_DIR, "xception_baseline.pth"),
    "LS-only": os.path.join(MODEL_DIR, "xception_ls_only.pth"),
    "JPEG-only": os.path.join(MODEL_DIR, "xception_jpeg_only.pth"),
    "JPEG+LS": os.path.join(MODEL_DIR, "xception_ls_jpeg.pth")
}

COMPRESS_QUALITY = 30
val_path = os.path.join(COMPRESSED_DIR, f"qf_{COMPRESS_QUALITY}")

print(f"验证集: {val_path}")

# ==================== 计算分离度 ====================
def compute_separation(features, labels):
    """计算类间分离度: 类间距离 / (类内距离 + 类间距离)"""
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    
    if len(real_feat) < 2 or len(fake_feat) < 2:
        return 0.0
    
    # 类内距离
    real_within = np.mean([np.linalg.norm(real_feat[i] - real_feat[j])
                           for i in range(len(real_feat))
                           for j in range(i+1, len(real_feat))])
    fake_within = np.mean([np.linalg.norm(fake_feat[i] - fake_feat[j])
                           for i in range(len(fake_feat))
                           for j in range(i+1, len(fake_feat))])
    within_dist = (real_within + fake_within) / 2
    
    # 类间距离
    between_dist = np.mean([np.linalg.norm(real_feat[i] - fake_feat[j])
                            for i in range(len(real_feat))
                            for j in range(len(fake_feat))])
    
    separation = between_dist / (within_dist + between_dist + 1e-8)
    return separation


def compute_5nn_accuracy(features, labels):
    """5-NN准确率"""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    preds = knn.predict(features)
    return accuracy_score(labels, preds)


def compute_model_accuracy(model_path, val_path):
    """模型在压缩集上的准确率"""
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

# ==================== 分析所有模型 ====================
all_metrics = []

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"⚠️ 模型不存在: {name}")
        continue
    
    print(f"\n分析 {name}...")
    
    # 1. 模型准确率
    acc = compute_model_accuracy(path, val_path)
    print(f"  模型准确率: {acc:.2%}")
    
    # 2. 特征提取
    features, labels = get_model_features(path, val_path, device, BATCH_SIZE)
    
    if features is None:
        continue
    
    # 3. 分离度
    separation = compute_separation(features, labels)
    print(f"  特征分离度: {separation:.4f}")
    
    # 4. 5-NN准确率
    acc_5nn = compute_5nn_accuracy(features, labels)
    print(f"  5-NN准确率: {acc_5nn:.2%}")
    
    all_metrics.append({
        "模型": name,
        "准确率": round(acc * 100, 2),
        "分离度": round(separation, 4),
        "5-NN准确率": round(acc_5nn * 100, 2)
    })

# ==================== 保存结果 ====================
df = pd.DataFrame(all_metrics)
df.to_csv(os.path.join(RESULT_DIR, "separation_metrics.csv"), index=False)

print("\n" + "="*60)
print("特征分离度量化结果")
print("="*60)
print(df.to_string(index=False))
print(f"\n✅ 结果已保存: {RESULT_DIR}/separation_metrics.csv")