from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip, Resize
from PIL import Image
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float
from skimage import feature
import cv2  # 新增OpenCV库
from sklearn.metrics import accuracy_score

# 定义Lee滤波（保持原实现）
def lee_filter(image: Image.Image):
    image_np = np.array(image)
    image_float = img_as_float(image_np)
    filtered_image = denoise_bilateral(image_float, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)
    return Image.fromarray((filtered_image * 255).astype(np.uint8))

# 修改后的Canny边缘检测（返回单通道图像）
def canny_edge_detection(image: Image.Image):
    # 转换为灰度图
    gray = image.convert("L")
    # 高斯滤波降噪
    gray_np = np.array(gray)
    blurred = cv2.GaussianBlur(gray_np, (5, 5), 0)
    # Canny检测
    edges = feature.canny(blurred, sigma=1.0, low_threshold=50, high_threshold=150)
    return Image.fromarray((edges * 255).astype(np.uint8))

# 数据预处理流程重构
def train_transforms(image: Image.Image):
    # 处理流程：Lee滤波 → 灰度 → Canny → 数据增强
    filtered = lee_filter(image)
    gray = filtered.convert("L")  # 转换为灰度
    edges = canny_edge_detection(gray)
    
    return Compose([
        RandomResizedCrop(processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ]) 

def val_transforms(image: Image.Image):
    filtered = lee_filter(image)
    gray = filtered.convert("L")
    edges = canny_edge_detection(gray)
    
    return Compose([
        Resize((processor.size["height"], processor.size["width"])),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ]) 

# 加载训练集和测试集
train_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-train",  # 训练集路径
    split="train"
)

test_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-test",   # 测试集路径
    split="train"
)

# 检查标签名称
print("Class names:", train_dataset.features["label"].names)

# 加载本地模型和处理器
model_path = "./VIT_model"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=10,
    ignore_mismatched_sizes=True  # 允许修改分类头尺寸
)

# 添加标签映射（可选，方便推理时显示类别名称）
model.config.id2label = {i: name for i, name in enumerate(train_dataset.features["label"].names)}
model.config.label2id = {name: i for i, name in enumerate(train_dataset.features["label"].names)}

# 应用预处理
def preprocess_train(examples):
    examples["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

# 预处理数据集
train_dataset = train_dataset.map(
    preprocess_train,
    batched=True,
    remove_columns=["image"]  # 移除原图列
)

test_dataset = test_dataset.map(
    preprocess_val,
    batched=True,
    remove_columns=["image"]
)

# 设置 PyTorch 张量格式
train_dataset.set_format("torch", columns=["pixel_values", "label"])
test_dataset.set_format("torch", columns=["pixel_values", "label"])

training_args = TrainingArguments(
    output_dir="./mstar_vit_output",  # 训练输出目录
    evaluation_strategy="epoch",       # 每个 epoch 评估一次
    save_strategy="epoch",             # 每个 epoch 是否保存模型
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
test_image = test_dataset[0]["pixel_values"].unsqueeze(0)  # 取第一张测试图
with torch.no_grad():
    logits = model(test_image).logits
predicted_class = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class]
print(f"Predicted label: {predicted_label}")