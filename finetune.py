import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from config import config
from models.vit_model import VisionTransformer
from utils.data_processing import create_kfold_loaders, load_raw_data
from utils.data_augmentation import get_train_transforms, MixUp
from utils.metrics import evaluate_model
from utils.visualization import plot_confusion_matrix, plot_classification_metrics, save_learning_curves

def parse_args():
    parser = argparse.ArgumentParser(description="ViT-based fault diagnosis finetuning")
    parser.add_argument("--processed_data_path", type=str, default=os.path.join(config.DATA_DIR, "processed_data.h5"),
                        help="Path to processed data")
    parser.add_argument("--pretrained_weights", type=str, default=os.path.join(config.OUTPUT_DIR, "pretrained_encoder.pth"),
                        help="Path to pretrained encoder weights")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr_backbone", type=float, default=config.FINETUNE_LR_BACKBONE,
                        help="Learning rate for backbone")
    parser.add_argument("--lr_head", type=float, default=config.FINETUNE_LR_HEAD,
                        help="Learning rate for classification head")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--k_folds", type=int, default=config.K_FOLDS,
                        help="Number of folds for cross-validation")
    return parser.parse_args()

def load_pretrained_weights(model, weights_path):
    """
    加载预训练权重
    
    Args:
        model: ViT模型
        weights_path: 预训练权重路径
    
    Returns:
        model: 加载预训练权重后的模型
    """
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights not found at {weights_path}. Training from scratch.")
        return model
    
    pretrained = torch.load(weights_path)
    
    # 加载Transformer块权重
    model_blocks = model.blocks.state_dict()
    pretrained_blocks = pretrained['blocks']
    
    # 检查键是否匹配
    model_keys = set(model_blocks.keys())
    pretrained_keys = set(pretrained_blocks.keys())
    
    # 只加载匹配的键
    common_keys = model_keys.intersection(pretrained_keys)
    
    # 创建新的状态字典
    new_state_dict = {}
    for key in common_keys:
        new_state_dict[key] = pretrained_blocks[key]
    
    # 加载Transformer块权重
    model.blocks.load_state_dict(new_state_dict, strict=False)
    
    # 加载类别标记和位置嵌入
    model.cls_token.data = pretrained['cls_token']
    model.pos_embed.data = pretrained['pos_embed']
    
    print(f"Loaded pretrained weights from {weights_path}")
    
    return model

def create_model(pretrained_weights=None):
    """
    创建模型
    
    Args:
        pretrained_weights: 预训练权重路径
    
    Returns:
        model: 创建的模型
    """
    # 初始化模型
    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.NUM_CHANNELS,
        n_classes=config.NUM_CLASSES,
        embed_dim=config.VIT_HIDDEN_SIZE,
        depth=config.VIT_NUM_LAYERS,
        n_heads=config.VIT_NUM_HEADS,
        mlp_ratio=config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE,
        drop_rate=config.DROPOUT_RATE
    )
    
    # 加载预训练权重
    if pretrained_weights:
        model = load_pretrained_weights(model, pretrained_weights)
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        use_mixup: 是否使用MixUp
    
    Returns:
        train_loss: 平均训练损失
        train_acc: 平均训练准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    mixup_fn = MixUp(alpha=0.2) if use_mixup else None
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 应用MixUp
        if use_mixup:
            inputs, labels_a, labels_b, lam = mixup_fn(inputs, labels)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算混合损失
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if use_mixup:
            # 混合标签的准确率计算
            correct += (lam * predicted.eq(labels_a).sum().float() 
                      + (1 - lam) * predicted.eq(labels_b).sum().float())
        else:
            correct += predicted.eq(labels).sum().item()
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        val_loss: 平均验证损失
        val_acc: 平均验证准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    
    return val_loss, val_acc

def main():
    args = parse_args()
    
    # 检查是否存在处理后的数据
    if not os.path.exists(args.processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {args.processed_data_path}. "
                               f"Run pretraining first to prepare data.")
    
    # 加载处理后的数据
    print(f"Loading processed data from {args.processed_data_path}...")
    with h5py.File(args.processed_data_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    
    print(f"Loaded processed data: {data.shape}, labels: {labels.shape}")
    
    # K折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\nFold {fold+1}/{args.k_folds}")
        
        # 划分数据
        train_data, train_labels = data[train_idx], labels[train_idx]
        val_data, val_labels = data[val_idx], labels[val_idx]
        
        # 创建数据加载器
        train_transforms = get_train_transforms()
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(train_data, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.long)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(val_data, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.long)
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        # 创建模型
        model = create_model(args.pretrained_weights)
        
        # 多GPU训练
        if config.NUM_GPUS > 1:
            model = nn.DataParallel(model)
        
        model = model.to(config.DEVICE)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 差异化学习率
        # 骨干网络使用较小的学习率
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # 定义优化器
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr_backbone},
            {'params': head_params, 'lr': args.lr_head}
        ], weight_decay=args.weight_decay)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.lr_backbone / 10
        )
        
        # 训练循环
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_val_acc = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # 训练
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = validate(
                model, val_loader, criterion, config.DEVICE
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(args.output_dir, f"best_model_fold{fold+1}.pth")
                if config.NUM_GPUS > 1:
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                print(f"Saved best model with val acc: {best_val_acc:.4f}")
        
        # 保存学习曲线
        save_dir = os.path.join(args.output_dir, f"fold{fold+1}_curves")
        save_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir)
        
        # 最终评估
        # 加载最佳模型
        model_path = os.path.join(args.output_dir, f"best_model_fold{fold+1}.pth")
        if config.NUM_GPUS > 1:
            model.module.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path))
        
        # 评估
        metrics = evaluate_model(model, val_loader, config.DEVICE)
        
        print(f"\nFold {fold+1} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['precision_macro']:.4f}")
        print(f"Macro Recall: {metrics['recall_macro']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        
        # 绘制混淆矩阵
        plt_path = os.path.join(args.output_dir, f"fold{fold+1}_confusion_matrix.png")
        plot_confusion_matrix(
            metrics['labels'],
            metrics['predictions'],
            figsize=(12, 10)
        )
        plt.savefig(plt_path)
        plt.close()
        
        # 保存结果
        fold_results.append({
            'fold': fold + 1,
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'best_val_acc': best_val_acc
        })
    
    # 打印所有折的平均结果
    print("\nAverage Results Across All Folds:")
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision_macro'] for r in fold_results])
    avg_recall = np.mean([r['recall_macro'] for r in fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
    
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Macro Precision: {avg_precision:.4f}")
    print(f"Macro Recall: {avg_recall:.4f}")
    print(f"Macro F1: {avg_f1:.4f}")
    
    # 保存最佳模型（以最高验证准确率为标准）
    best_fold = max(fold_results, key=lambda x: x['accuracy'])['fold']
    best_model_path = os.path.join(args.output_dir, f"best_model_fold{best_fold}.pth")
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    
    # 复制最佳模型
    import shutil
    shutil.copy(best_model_path, final_model_path)
    print(f"\nBest model from fold {best_fold} saved as final model.")

if __name__ == "__main__":
    main()