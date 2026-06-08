## 📌 项目简介

本项目针对社交媒体压缩导致Deepfake检测性能下降的问题，提出了Label Smoothing + JPEG压缩增强的组合训练策略。
This project addresses the performance degradation of Deepfake detection caused by social media compression, proposing a combined training strategy of **Label Smoothing + JPEG Augmentation**.

**核心贡献：**
- ✅ 验证Xception在压缩场景下的性能退化（99.30% → 61.30%）
  *Validated Xception's performance degradation under compression (99.30% → 61.30%)*
  
- ✅ 通过消融实验证明单一策略无效，两者协同有效
  *Ablation study proves single strategies ineffective, combination works*
  
- ✅ 改进模型在QF=30时达**72.71%**，相比基线提升11.41个百分点
  *Improved model achieves **72.71%** at QF=30, +11.41 percentage points over baseline*

---

## 📊 核心实验结果

### 压缩鲁棒性对比（QF=30）Compression Robustness (QF=30)

| 模型 | 准确率 | 下降幅度 |
|------|--------|----------|
| Baseline | 61.30% | 38.0% |
| JPEG Only | 55.04% | 44.18% |
| Label Smoothing | 58.01% | 40.90% |
| **LS + JPEG (Ours)** | **72.71%** | **25.96%** |

### 特征分离度分析（QF=30）Feature Separability (QF=30)

| 模型 | 分离度 | 5-NN准确率 |
|------|--------|-----------|
| Baseline | 0.4053 | 81.43% |
| JPEG Only | 0.3900 | 77.14% |
| Label Smoothing | 0.5060 | 88.57% |
| **LS + JPEG (Ours)** | **0.5498** | **87.14%** |

### 泛化实验 Generalization

| 测试集 | Baseline | LS+JPEG |
|--------|----------|---------|
| 中国人脸（GAN架构） | 85.16% | 84.04% |
| CollabDiff（扩散模型） | 33.00% | 33.10% |

---

## 📈 可视化结果 Visualizations

| 压缩实验 | 基线崩塌实验 |
| Compression Experiment | Baseline Collapse |
| 压缩性能对比 | 特征分布(t-SNE) |
| Performance Comparison | Feature Distribution |
| 置信度分析 | 消融实验 |
| Confidence Analysis | Ablation Study |

## ⭐ Star

如果这个项目对你有帮助，请给一个Star ⭐
*If this project helps you, please give it a Star ⭐*
