import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import torch
import seaborn as sns
from config import config
import os

def plot_stft(stft_data, title=None):
    """
    绘制STFT时频图
    
    Args:
        stft_data: STFT数据
        title: 图表标题
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(stft_data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, figsize=(12, 10), ax=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        figsize: 图像大小
        ax: matplotlib轴对象
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图像和轴
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # 绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # 设置标签
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    return ax

def plot_classification_metrics(y_true, y_pred, figsize=(15, 8)):
    """
    绘制分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 获取分类报告
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 提取每个类别的指标
    classes = []
    precision = []
    recall = []
    f1_score = []
    
    for key, value in report.items():
        if key.isdigit():
            classes.append(int(key))
            precision.append(value['precision'])
            recall.append(value['recall'])
            f1_score.append(value['f1-score'])
    
    # 排序
    sorted_indices = np.argsort(classes)
    classes = np.array(classes)[sorted_indices]
    precision = np.array(precision)[sorted_indices]
    recall = np.array(recall)[sorted_indices]
    f1_score = np.array(f1_score)[sorted_indices]
    
    # 绘制条形图
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1 Score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Classification Metrics')
    plt.xticks(x, classes)
    plt.legend()
    
    return plt.gca()

def plot_feature_space(features, labels, figsize=(12, 10)):
    """
    使用t-SNE绘制特征空间
    
    Args:
        features: 特征向量
        labels: 标签
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 降维
    print("计算t-SNE降维...")
    tsne = TSNE(n_components=2, 
            perplexity=30, 
            n_iter=1000,
            random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # 绘制散点图
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7
        )
    
    plt.title('t-SNE Feature Visualization')
    plt.legend()
    plt.tight_layout()
    
    return plt.gca()

def visualize_attention(model, x, head_idx=0, layer_idx=-1, save_path=None):
    """
    可视化注意力图
    
    Args:
        model: ViT模型
        x: 输入图像
        head_idx: 注意力头索引
        layer_idx: 层索引
        save_path: 保存路径
    """
    model.eval()
    
    # 如果x是numpy数组，转换为tensor
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
        
    # 添加批次维度如果需要
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    # 获取模型配置
    if hasattr(model, 'module'):
        patch_size = model.module.patch_embed.patch_size
    else:
        patch_size = model.patch_embed.patch_size
        
    # 注册钩子
    attention_maps = []
    
    def hook_fn(module, input, output):
        # 对于Transformer的注意力模块，输出通常是(batch_size, num_heads, seq_len, seq_len)
        attention_maps.append(output)
    
    # 根据模型结构注册钩子 - 增加错误处理
    try:
        if hasattr(model, 'module'):
            # DataParallel包装的模型
            if hasattr(model.module.blocks[layer_idx].attn, 'attn_drop'):
                attn_module = model.module.blocks[layer_idx].attn.attn_drop
            elif hasattr(model.module.blocks[layer_idx].attn, 'attn'):
                attn_module = model.module.blocks[layer_idx].attn.attn
            else:
                attn_module = model.module.blocks[layer_idx].attn
        else:
            # 普通模型
            if hasattr(model.blocks[layer_idx].attn, 'attn_drop'):
                attn_module = model.blocks[layer_idx].attn.attn_drop
            elif hasattr(model.blocks[layer_idx].attn, 'attn'):
                attn_module = model.blocks[layer_idx].attn.attn
            else:
                attn_module = model.blocks[layer_idx].attn
        
        hook = attn_module.register_forward_hook(hook_fn)
    except Exception as e:
        print(f"注册钩子时出错: {e}")
        print("尝试使用备用方法...")
        # 备用方法：找到包含'attn'的任何模块
        for name, module in model.named_modules():
            if 'attn' in name and 'drop' in name:
                print(f"使用模块: {name}")
                hook = module.register_forward_hook(hook_fn)
                break
        else:
            for name, module in model.named_modules():
                if 'attn' in name:
                    print(f"使用模块: {name}")
                    hook = module.register_forward_hook(hook_fn)
                    break
            else:
                print("无法找到合适的注意力模块")
                return None
    
    # 前向传播，但不计算梯度
    try:
        with torch.no_grad():
            _ = model(x.to(config.DEVICE))
        
        # 移除钩子
        hook.remove()
        
        # 检查是否成功获取了注意力图
        if not attention_maps:
            print("未能获取注意力图")
            return None
        
        # 获取注意力图
        attention = attention_maps[0]
        
        # 灵活处理不同形状的注意力图
        if len(attention.shape) == 4:
            # 形状为 [batch, heads, seq, seq]
            if attention.shape[1] > head_idx:
                attention_map = attention[0, head_idx].cpu()
            else:
                print(f"头索引 {head_idx} 超出范围 (0-{attention.shape[1]-1})")
                attention_map = attention[0, 0].cpu()
        elif len(attention.shape) == 3:
            # 形状为 [batch, seq, seq]
            attention_map = attention[0].cpu()
        else:
            print(f"无法处理形状为 {attention.shape} 的注意力张量")
            if torch.is_tensor(attention):
                # 尝试转换为可视化格式
                attention_map = attention.view(-1).reshape(int(np.sqrt(attention.numel())), 
                                                        int(np.sqrt(attention.numel()))).cpu()
            else:
                return None
        
        # 处理注意力图，如果形状合适，移除cls_token
        if attention_map.shape[0] > 1 and attention_map.shape[0] == attention_map.shape[1]:
            try:
                attention_map = attention_map[1:, 1:]
            except Exception:
                pass  # 如果出错则保持原样
        
        # 重塑成2D网格
        img_size = config.IMG_SIZE
        try:
            if isinstance(patch_size, tuple):
                h_patches = img_size // patch_size[0]
                w_patches = img_size // patch_size[1]
                
                # 确保尺寸匹配
                if attention_map.numel() == h_patches * w_patches:
                    attention_map = attention_map.reshape(h_patches, w_patches)
                else:
                    # 尝试使用合适的网格尺寸
                    grid_size = int(np.sqrt(attention_map.numel()))
                    attention_map = attention_map.reshape(grid_size, grid_size)
            else:
                # 当patch_size为整数时
                num_patches = (img_size // patch_size) ** 2
                
                # 确保尺寸匹配
                if attention_map.numel() == num_patches:
                    grid_size = int(np.sqrt(num_patches))
                    attention_map = attention_map.reshape(grid_size, grid_size)
                else:
                    # 尝试使用合适的网格尺寸
                    grid_size = int(np.sqrt(attention_map.numel()))
                    attention_map = attention_map.reshape(grid_size, grid_size)
        except Exception as e:
            print(f"重塑注意力图时出错: {e}")
            # 尝试使用最简单的方式重塑
            total_elements = attention_map.numel()
            grid_size = int(np.sqrt(total_elements))
            attention_map = attention_map.flatten()[:grid_size*grid_size].reshape(grid_size, grid_size)
        
        # 绘制注意力图
        plt.figure(figsize=(10, 10))
        
        # 首先绘制输入图像
        plt.subplot(1, 2, 1)
        # 选择第一个通道作为可视化示例
        plt.imshow(x[0, 0].cpu(), cmap='viridis')
        plt.title('Input Image (Channel 0)')
        plt.axis('off')
        
        # 然后绘制注意力图
        plt.subplot(1, 2, 2)
        plt.imshow(attention_map, cmap='hot', interpolation='nearest')
        plt.title(f'Layer {layer_idx}, Head {head_idx} Attention Map')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path)
            plt.close()
        
        return plt.gcf()
    
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        if 'hook' in locals():
            hook.remove()
        return None
    
def save_learning_curves(train_losses, val_losses, train_accs=None, val_accs=None, save_dir='./plots'):
    """
    保存学习曲线
    
    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率
        val_accs: 验证准确率
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
        # 确保所有张量移至CPU并转换为NumPy
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor
    
    train_losses = to_numpy(train_losses)
    val_losses = to_numpy(val_losses)
    train_accs = to_numpy(train_accs)
    val_accs = to_numpy(val_accs)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # 如果提供了准确率，绘制准确率曲线
    if train_accs and val_accs:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
        plt.close()