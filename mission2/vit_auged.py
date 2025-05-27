from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip, Resize, RandomRotation, RandomAffine, Lambda
from PIL import Image, ImageFilter
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import cv2

# acc : 0.9386

# 加载训练集和测试集
train_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-train",  # 训练集路径
    split="train"      # 返回完整数据集
)

test_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-test",   # 测试集路径
    split="train"   # 返回完整数据
)

# 检查标签名称
print("Class names:", train_dataset.features["label"].names)

# 加载本地模型和处理器
model_path = "./VIT_model"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=10,  # 根据您的10类任务调整
    ignore_mismatched_sizes=True  # 允许修改分类头尺寸
)



# 定义数据增强组件
def sar_noise_injection(image):
    """SAR噪声模拟三通道版本"""
    np_img = np.array(image).astype(np.float32)/255.0
    # 生成三通道噪声（与输入维度一致）
    noise = np.random.gamma(shape=3, scale=0.2, size=np_img.shape)
    noisy_img = np.clip(np_img * noise * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 修改后的边缘检测函数（保持输入维度）
def generate_edge_map(image):
    """边缘检测（适配任意输入模式）"""
    np_img = np.array(image)
    # 如果是彩色图取第一个通道
    if len(np_img.shape) == 3:
        np_img = np_img[:,:,0]
    edges = cv2.Canny(np_img, 30, 80)
    return Image.fromarray(edges)

# 修正后的训练数据转换
def train_transforms(image: Image.Image):
    # 三组增强策略
    return [
        # 第一组：多模态融合（双通道+噪声）
        Compose([
            RandomResizedCrop(processor.size["height"], scale=(0.7,1.3)),
            Lambda(lambda x: x.convert("L")),  # 统一转换为灰度
            Lambda(lambda x: Image.merge("RGB", [  # 三通道构造
                x,  # 原始通道
                generate_edge_map(x), 
                sar_noise_injection(x)
            ])),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ])(image),
        
        # 第二组：几何变换增强
        Compose([
            RandomResizedCrop(processor.size["height"], scale=(0.8,1.2)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(360),
            RandomAffine(degrees=0, translate=(0.1,0.1)),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ])(image),
        
        # 第三组：传统增强组合
        Compose([
            RandomResizedCrop(processor.size["height"], scale=(0.9,1.1)),
            Lambda(lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)) if random.random()>0.5 else x),
            Lambda(lambda x: sar_noise_injection(x)),  # 基于处理后的x生成
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ])(image)
    ]

def val_transforms(image: Image.Image):
    return Compose([
        Resize((processor.size["height"], processor.size["width"])),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])(image)

# 应用预处理
def preprocess_train(examples):
    all_pixels = []
    all_labels = []
    for img, label in zip(examples["image"], examples["label"]):
        # 生成三种增强版本
        augmented_images = train_transforms(img.convert("RGB"))
        all_pixels.extend(augmented_images)
        all_labels.extend([label]*3)  # 每个样本生成三个增强样本
    
    return {"pixel_values": all_pixels, "label": all_labels}

def preprocess_val(examples):
    examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

# 预处理数据集
train_dataset = train_dataset.map(
    preprocess_train,
    batched=True,
    remove_columns=["image"]
)

test_dataset = test_dataset.map(
    preprocess_val,
    batched=True,
    remove_columns=["image"]
)

# 设置PyTorch格式
train_dataset.set_format("torch", columns=["pixel_values", "label"])
test_dataset.set_format("torch", columns=["pixel_values", "label"])

training_args = TrainingArguments(
    output_dir="./mstar_vit_output",  # 训练输出目录
    evaluation_strategy="no",       # 每个 epoch 评估一次
    save_strategy="no",             # 每个 epoch 保存模型
    learning_rate=2e-5,                # 学习率
    per_device_train_batch_size=32,    # 根据 GPU 内存调整
    per_device_eval_batch_size=32,
    num_train_epochs=10,               # 训练轮数
    logging_steps=50,
    load_best_model_at_end=True,       # 训练结束时加载最优模型
    metric_for_best_model="accuracy",
    fp16=True,                         # 启用混合精度训练（需 GPU 支持）
    report_to="none"                   # 禁用第三方日志（如 wandb）
)



def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
train_results = trainer.train()

# 保存最佳模型
trainer.save_model("./mstar_vit_finetuned")

# 加载保存的最佳模型
model = ViTForImageClassification.from_pretrained("./mstar_vit_finetuned")

# 在测试集上评估
eval_results = trainer.evaluate(test_dataset)
print(f"Final Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# 单张图片预测
from PIL import Image

test_image = test_dataset[0]["pixel_values"].unsqueeze(0)  # 取第一张测试图
with torch.no_grad():
    logits = model(test_image).logits
predicted_class = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class]
print(f"Predicted label: {predicted_label}")