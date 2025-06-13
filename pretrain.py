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

from config import config
from models.pretrain_model import MaskedAutoencoderViT
from utils.data_processing import prepare_dataset, create_data_loaders
from utils.data_augmentation import get_train_transforms
from utils.visualization import plot_stft, save_learning_curves

def parse_args():
    parser = argparse.ArgumentParser(description="ViT-based fault diagnosis pretraining")
    parser.add_argument("--data_path", type=str, default=os.path.join(config.DATA_DIR, "raw_data.h5"),
                        help="Path to raw data")
    parser.add_argument("--processed_data_path", type=str, default=os.path.join(config.DATA_DIR, "processed_data.h5"),
                        help="Path to save processed data")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=config.PRETRAIN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.PRETRAIN_LR,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--mask_ratio", type=float, default=config.MASK_RATIO,
                        help="Ratio of masked patches")
    return parser.parse_args()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for inputs, _ in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        pred, mask, _ = model(inputs)
        
        # 计算重建损失
        loss = compute_reconstruction_loss(inputs, pred, mask, model)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def compute_reconstruction_loss(inputs, pred, mask, model):
    """
    计算重建损失
    
    Args:
        inputs: 原始输入
        pred: 模型预测
        mask: 掩码
        model: 模型
    
    Returns:
        loss: 重建损失
    """
    # 获取patch大小
    patch_size = model.patch_embed.patch_size
    
    # 将输入重塑为patch形式
    B, C, H, W = inputs.shape
    p = patch_size
    h, w = H // p, W // p
    target = inputs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*p*p)
    
    # 计算掩码patch的损失
    loss = F.mse_loss(pred[:, 1:], target, reduction='none')  # [B, N, C*P*P]
    loss = loss.mean(dim=-1)  # [B, N]
    
    # 只考虑被掩码的patch
    mask = mask.to(inputs.device)
    loss = (loss * mask).sum() / mask.sum()
    
    return loss

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            
            # 前向传播
            pred, mask, _ = model(inputs)
            
            # 计算重建损失
            loss = compute_reconstruction_loss(inputs, pred, mask, model)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    args = parse_args()
    
    # 检查是否已处理数据
    if not os.path.exists(args.processed_data_path):
        print(f"Processing data from {args.data_path}...")
        prepare_dataset(args.data_path, args.processed_data_path)
    
    # 加载处理后的数据
    with h5py.File(args.processed_data_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    
    print(f"Loaded processed data: {data.shape}, labels: {labels.shape}")
    
    # 划分训练和验证集（90%/10%）
    indices = np.random.permutation(len(data))
    train_size = int(0.9 * len(data))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data, train_labels = data[train_indices], labels[train_indices]
    val_data, val_labels = data[val_indices], labels[val_indices]
    
    # 创建数据加载器
    train_transforms = get_train_transforms()
    train_loader = create_data_loaders(train_data, train_labels, args.batch_size, train_transforms)
    val_loader = create_data_loaders(val_data, val_labels, args.batch_size)
    
    # 初始化模型
    model = MaskedAutoencoderViT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.NUM_CHANNELS,
        embed_dim=config.VIT_HIDDEN_SIZE,
        depth=config.VIT_NUM_LAYERS,
        n_heads=config.VIT_NUM_HEADS,
        decoder_embed_dim=config.VIT_HIDDEN_SIZE // 2,
        decoder_depth=6,
        decoder_n_heads=config.VIT_NUM_HEADS // 2,
        mlp_ratio=config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE,
        mask_ratio=args.mask_ratio
    )
    
    # 多GPU训练
    if config.NUM_GPUS > 1:
        model = nn.DataParallel(model)
    
    model = model.to(config.DEVICE)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr / 100
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train(model, train_loader, optimizer, config.DEVICE)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, config.DEVICE)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存模型权重（只保存编码器部分）
            if config.NUM_GPUS > 1:
                encoder_state_dict = model.module.blocks.state_dict()
                cls_token = model.module.cls_token.data
                pos_embed = model.module.pos_embed.data
            else:
                encoder_state_dict = model.blocks.state_dict()
                cls_token = model.cls_token.data
                pos_embed = model.pos_embed.data
            
            # 保存编码器权重
            torch.save({
                'blocks': encoder_state_dict,
                'cls_token': cls_token,
                'pos_embed': pos_embed,
                'img_size': config.IMG_SIZE,
                'patch_size': config.PATCH_SIZE,
                'in_channels': config.NUM_CHANNELS,
                'embed_dim': config.VIT_HIDDEN_SIZE,
                'depth': config.VIT_NUM_LAYERS,
                'n_heads': config.VIT_NUM_HEADS,
                'mlp_ratio': config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE
            }, os.path.join(args.output_dir, 'pretrained_encoder.pth'))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
    
    # 保存学习曲线
    save_learning_curves(train_losses, val_losses, [], [], os.path.join(args.output_dir, 'pretrain_curves'))
    
    print(f"Pretraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()