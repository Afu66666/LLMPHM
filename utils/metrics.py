import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config import config

def evaluate_model(model, data_loader, device=config.DEVICE):
    """
    评估模型性能
    
    Args:
        model: 模型
        data_loader: 测试数据加载器
        device: 设备
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="评估"):
            data, labels = data.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(data)
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels
    }

def extract_features(model, data_loader, device=config.DEVICE, layer='cls_token'):
    """
    提取模型特征
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        layer: 特征提取层
        
    Returns:
        features: 特征向量
        labels: 对应标签
    """
    model.eval()
    features = []
    all_labels = []
    
    # 注册钩子函数
    if layer == 'cls_token':
        feature_layer = model
        if hasattr(model, 'module'):
            feature_layer = model.module
            
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output[:, 0].detach().cpu().numpy()  # 获取CLS token
            return hook
        
        # 注册钩子到头部前的最后一层
        if hasattr(feature_layer, 'norm'):
            handle = feature_layer.norm.register_forward_hook(get_activation('cls_token'))
        else:
            print("警告: 模型中没有找到norm层，尝试直接使用输出")
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="提取特征"):
            data, labels = data.to(device), labels.to(device)
            
            # 前向传播
            _ = model(data)
            
            if layer == 'cls_token' and 'cls_token' in activation:
                batch_features = activation['cls_token']
                features.append(batch_features)
            else:
                print("警告: 没有找到CLS token，使用默认输出")
                outputs = model(data)
                batch_features = outputs.detach().cpu().numpy()
                features.append(batch_features)
                
            all_labels.extend(labels.cpu().numpy())
    
    # 移除钩子
    if layer == 'cls_token' and 'handle' in locals():
        handle.remove()
    
    # 合并所有批次的特征
    features = np.vstack(features)
    all_labels = np.array(all_labels)
    
    return features, all_labels

def compute_loss(criterion, outputs, labels, mixup=False, labels_a=None, labels_b=None, lam=None):
    """
    计算损失函数
    
    Args:
        criterion: 损失函数
        outputs: 模型输出
        labels: 标签
        mixup: 是否使用mixup
        labels_a: mixup标签A
        labels_b: mixup标签B
        lam: mixup混合权重
        
    Returns:
        loss: 损失值
    """
    if mixup:
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    else:
        loss = criterion(outputs, labels)
    return loss

def compute_reconstruction_loss(inputs, pred, mask, model):
    """
    计算MAE重建损失
    
    Args:
        inputs: 输入数据
        pred: 模型预测输出
        mask: 掩码
        model: MAE模型
        
    Returns:
        loss: 重建损失
    """
    # 获取patch大小
    if hasattr(model, 'module'):  # DataParallel包装
        patch_size = model.module.patch_embed.patch_size
    else:
        patch_size = model.patch_embed.patch_size
    
    # 将输入重塑为patch形式
    B, C, H, W = inputs.shape
    p = patch_size
    h, w = H // p, W // p
    target = inputs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*p*p)
    
    # 计算掩码patch的损失
    loss = torch.nn.functional.mse_loss(pred[:, 1:], target, reduction='none')  # [B, N, C*P*P]
    loss = loss.mean(dim=-1)  # [B, N]
    
    # 只考虑被掩码的patch
    mask = mask.to(inputs.device)
    loss = (loss * mask).sum() / mask.sum()
    
    return loss