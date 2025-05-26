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

# 加载原始训练集和测试集
original_train_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-train",
    split="train"
)

test_dataset = load_dataset(
    "imagefolder",
    data_dir="./MSTAR/mstar-test",
    split="train"
)

# 获取初始的标记数据索引
def get_subsample_indices(dataset, samples_per_class=10):
    labels = dataset["label"]
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    selected_indices = []
    for label in class_indices:
        class_samples = class_indices[label]
        selected = random.sample(class_samples, samples_per_class)
        selected_indices.extend(selected)
    return selected_indices

random.seed(42)
labeled_indices = get_subsample_indices(original_train_dataset, 10)
unlabeled_indices = [i for i in range(len(original_train_dataset)) if i not in labeled_indices]

# 创建标记和未标记数据集
labeled_dataset = original_train_dataset.select(labeled_indices)
unlabeled_dataset = original_train_dataset.select(unlabeled_indices)

print("Initial labeled dataset size:", len(labeled_dataset))
print("Initial unlabeled dataset size:", len(unlabeled_dataset))
print("Class names:", labeled_dataset.features["label"].names)

# 加载本地模型和处理器
model_path = "./VIT_model"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=10,
    ignore_mismatched_sizes=True
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

# 初始预处理标记数据集
labeled_processed = labeled_dataset.map(
    preprocess_train,
    batched=True,
    remove_columns=["image"]
)
labeled_processed.set_format("torch", columns=["pixel_values", "label"])

test_dataset= test_dataset.map(
    preprocess_val,
    batched=True,
    remove_columns=["image"]
)
test_dataset.set_format("torch", columns=["pixel_values", "label"])

# 训练参数
training_args = TrainingArguments(
    output_dir="./mstar_vit_output_self_training",
    evaluation_strategy="no",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none"
)


# 计算指标
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

# 初始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=labeled_processed,
    compute_metrics=compute_metrics,
)
trainer.train()

# 自训练参数
num_iterations = 3
# confidence_threshold = 0.35
start_threshold = 0.35  # 初始阈值
end_threshold = 0.95    # 最终阈值 

# 自训练循环
for iteration in range(num_iterations):
    print(f"\nSelf-training iteration {iteration + 1}/{num_iterations}")

    # 计算当前迭代的置信度阈值
    if num_iterations > 1:
        progress = iteration / (num_iterations - 1)
        current_confidence_threshold = start_threshold + (end_threshold - start_threshold) * progress
    else: # 处理 num_iterations = 1 的情况
        current_confidence_threshold = start_threshold
    
    # 预处理未标记数据集用于预测
    unlabeled_processed = unlabeled_dataset.map(
        preprocess_val,
        batched=True,
        remove_columns=["image"]
    )
    unlabeled_processed.set_format("torch", columns=["pixel_values"])
    
    # 预测伪标签
    temp_trainer = Trainer(model=model)
    predictions = temp_trainer.predict(unlabeled_processed)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    
    # 选择高置信度样本
    high_conf_indices = []
    high_conf_labels = []
    for i, prob in enumerate(probs):
        max_prob = np.max(prob)
        if max_prob > current_confidence_threshold:
            high_conf_indices.append(i)
            high_conf_labels.append(np.argmax(prob))
    
    if not high_conf_indices:
        print("No high-confidence samples found. Stopping self-training.")
        break
    
    # 获取全局索引
    selected_global_indices = [unlabeled_indices[i] for i in high_conf_indices]
    
    # 更新索引
    labeled_indices.extend(selected_global_indices)
    unlabeled_indices = [i for i in unlabeled_indices if i not in selected_global_indices]
    
    # 更新数据集
    labeled_dataset = original_train_dataset.select(labeled_indices)
    unlabeled_dataset = original_train_dataset.select(unlabeled_indices)
    
    print(f"Added {len(selected_global_indices)} new samples. Labeled dataset size: {len(labeled_dataset)}")
    
    # 重新预处理标记数据集
    labeled_processed = labeled_dataset.map(
        preprocess_val,
        batched=True,
    )
    labeled_processed.set_format("torch", columns=["pixel_values", "label"])
    
    # 重新训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_processed,
        compute_metrics=compute_metrics,
    )
    trainer.train()

# 最终评估
eval_results = trainer.evaluate(test_dataset)
print(f"Final Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# 保存模型
trainer.save_model("./mstar_vit_self_trained")
model = ViTForImageClassification.from_pretrained("./mstar_vit_self_trained")

# # 单样本预测示例
# test_image = test_dataset[0]["pixel_values"].unsqueeze(0)
# plt.imshow(test_image[0].permute(1, 2, 0).numpy())
# plt.axis("off")
# plt.show()
# with torch.no_grad():
#     logits = model(test_image).logits
# predicted_class = logits.argmax(-1).item()
# print(f"Predicted label: {model.config.id2label[predicted_class]}")