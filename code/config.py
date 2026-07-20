"""
config.py - 统一路径配置
"""

import os
import torch
from torchvision import transforms

# 数据集路径
BASE_DIR = r"D:\lunwen 1\Deepfake_Xception_Exp"

TRAIN_PATH = os.path.join(BASE_DIR, "datasets", "train")
VAL_PATH = os.path.join(BASE_DIR, "datasets", "val")
MODEL_DIR = os.path.join(BASE_DIR, "scripts", "模型")
COMPRESSED_DIR = os.path.join(BASE_DIR, "dataset_compressed")
RESULT_DIR = os.path.join(BASE_DIR, "scripts", "结果")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(COMPRESSED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 24
LEARNING_RATE = 1e-4

# 数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

print(f"使用设备: {device}")
print(f"训练集: {TRAIN_PATH}")
print(f"验证集: {VAL_PATH}")
print(f"模型保存: {MODEL_DIR}")
