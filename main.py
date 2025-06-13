
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from config import config
from utils.data_cache import cache_exists, load_from_cache, save_to_cache, get_cache_path
from utils.data_loader import DataLoader as CustomDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="基于Vision Transformer的故障诊断流程")
    parser.add_argument("--mode", type=str, default="all", choices=["preprocess", "pretrain", "finetune", "evaluate", "all"],
                        help="运行模式")
    parser.add_argument("--data_path", type=str, default="Datasets",
                        help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--pretrained_weights", type=str, default=os.path.join(config.OUTPUT_DIR, "pretrained_encoder.pth"),
                        help="预训练权重路径")
    parser.add_argument("--final_model_path", type=str, default=os.path.join(config.OUTPUT_DIR, "final_model.pth"),
                        help="最终模型路径")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="批次大小")
    parser.add_argument("--pretrain_epochs", type=int, default=config.PRETRAIN_EPOCHS,
                        help="预训练轮数")
    parser.add_argument("--finetune_epochs", type=int, default=config.FINETUNE_EPOCHS,
                        help="微调轮数")
    parser.add_argument("--k_folds", type=int, default=config.K_FOLDS,
                        help="交叉验证折数")
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="是否使用缓存数据")
    parser.add_argument("--visualize", action="store_true",
                        help="可视化样本")
    return parser.parse_args()

def preprocess(args):
    """数据预处理"""
    print("\n" + "="*50)
    print("开始数据预处理...")
    
    from utils.data_processing import prepare_dataset, prepare_test_dataset
    from utils.visualization import plot_stft
    
    # 检查数据路径
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据目录未找到: {args.data_path}")
    
    # 处理训练数据
    train_data, train_labels, train_conditions = prepare_dataset(
        args.data_path, 
        use_cache=args.use_cache
    )
    
    # 处理测试数据
    test_data, test_labels, test_conditions = prepare_test_dataset(
        args.data_path,
        use_cache=args.use_cache
    )
    
    print(f"数据预处理完成")
    print(f"训练数据形状: {train_data.shape}, 标签形状: {train_labels.shape}")
    print(f"测试数据形状: {test_data.shape}, 标签形状: {test_labels.shape}")
    print("="*50)
    
    # 可视化一些样本
    if args.visualize:
        # 可视化几个样本
        num_samples = min(5, len(train_data))
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(train_data[i][0], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"样本 {i+1}, 类别 {train_labels[i]+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sample_visualizations.png"))
        plt.close()

def pretrain(args):
    """模型预训练"""
    print("\n" + "="*50)
    print("开始模型预训练...")
    
    # 导入预训练所需模块
    from models.pretrain_model import MaskedAutoencoderViT
    from utils.data_processing import prepare_dataset, create_data_loaders
    from utils.data_augmentation import get_train_transforms
    import torch.optim as optim
    
    # 加载处理后的数据
    train_data, train_labels, _ = prepare_dataset(
        args.data_path, 
        use_cache=args.use_cache
    )
    print(f"训练数据形状: {train_data.shape}, 标签形状: {train_labels.shape}")
    print("="*50)
    from sklearn.model_selection import train_test_split

    # 使用分层抽样，保证每个类别在训练集和验证集中的比例一致
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, 
        train_labels, 
        test_size=0.2,                # 验证集占20%
        random_state=42,              # 设置随机种子确保结果可复现
        stratify=train_labels         # 关键参数：确保按类别标签分层抽样
    )
    
    # 创建数据加载器
    train_transforms = get_train_transforms()
    train_loader = create_data_loaders(train_data, train_labels, args.batch_size, train_transforms)
    val_loader = create_data_loaders(val_data, val_labels, args.batch_size)
    
    # 初始化预训练模型
    num_channels = train_data.shape[1]  # 获取通道数
    model = MaskedAutoencoderViT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=num_channels,
        embed_dim=config.VIT_HIDDEN_SIZE,
        depth=config.VIT_NUM_LAYERS,
        n_heads=config.VIT_NUM_HEADS,
        decoder_embed_dim=config.VIT_HIDDEN_SIZE // 2,
        decoder_depth=6,
        decoder_n_heads=config.VIT_NUM_HEADS // 2,
        mlp_ratio=config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE,
        mask_ratio=config.MASK_RATIO
    )
    
    # 多GPU训练
    if config.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(config.DEVICE)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.PRETRAIN_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.pretrain_epochs, 
        eta_min=config.PRETRAIN_LR / 100
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 定义重建损失计算函数
    def compute_reconstruction_loss(inputs, pred, mask, model):
        # 获取patch大小
        if hasattr(model, 'module'):  # DataParallel包装
            patch_size = model.module.patch_embed.patch_size
        else:
            patch_size = model.patch_embed.patch_size
        
        # 将输入重塑为patch形式
        B, C, H, W = inputs.shape
        p = patch_size
        
        p_h = p[0] if isinstance(p, tuple) else p
        p_w = p[1] if isinstance(p, tuple) else p
        h, w = H // p_h, W // p_w
        target = inputs.reshape(B, C, h, p_h, w, p_w).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*p_h*p_w)
        
        # 计算掩码patch的损失
        loss = torch.nn.functional.mse_loss(pred, target, reduction='none')  # [B, N, C*P*P]
        loss = loss.mean(dim=-1)  # [B, N]
        
        # 只考虑被掩码的patch
        mask = mask.to(inputs.device)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    for epoch in range(args.pretrain_epochs):
        print(f"\nEpoch {epoch+1}/{args.pretrain_epochs}")
        
        # 训练
        model.train()
        total_train_loss = 0
        
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE, non_blocking=True)  # 添加 non_blocking=True
            
            optimizer.zero_grad()
            # 前向传播
            pred, mask, _ = model(inputs)
            loss = compute_reconstruction_loss(inputs, pred, mask, model)
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        total_val_loss = 0
                
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(config.DEVICE, non_blocking=True)
                # 前向传播
                pred, mask, _ = model(inputs)
                
                # 计算重建损失
                loss = compute_reconstruction_loss(inputs, pred, mask, model)
                
                # 减少CPU-GPU同步，直接累积tensor
                total_val_loss += loss
            
            # 在循环结束后只调用一次item()
            val_loss = (total_val_loss / len(val_loader)).item()

        # 记录验证损失
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step()

        # 简化输出
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

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
                'in_channels': num_channels,
                'embed_dim': config.VIT_HIDDEN_SIZE,
                'depth': config.VIT_NUM_LAYERS,
                'n_heads': config.VIT_NUM_HEADS,
                'mlp_ratio': config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE
            }, args.pretrained_weights)
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
            
    # 保存预训练结果到缓存
    from utils.visualization import save_learning_curves
    save_learning_curves(train_losses, val_losses, [], [], os.path.join(args.output_dir, 'pretrain_curves'))
    
    print(f"预训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print("="*50)

def finetune(args):
    """模型微调"""
    print("\n" + "="*50)
    print("开始模型微调...")
    
    # 导入微调所需模块
    from models.vit_model import VisionTransformer
    from utils.data_processing import prepare_dataset, create_kfold_loaders
    from utils.data_augmentation import get_train_transforms, MixUp
    from utils.metrics import evaluate_model
    import torch.optim as optim
    import torch.nn as nn
    from tqdm import tqdm
    from utils.visualization import plot_confusion_matrix, plot_classification_metrics, save_learning_curves
    
    # 加载处理后的数据
    data, labels, _ = prepare_dataset(
        args.data_path, 
        use_cache=args.use_cache
    )
    print(f"训练数据形状: {data.shape}, 标签形状: {labels.shape}")
    print("="*50)
    # 检查预训练权重
    if not os.path.exists(args.pretrained_weights):
        print(f"警告: 预训练权重未找到: {args.pretrained_weights}. 将从头开始训练。")
        pretrained_weights = None
    else:
        pretrained_weights = args.pretrained_weights
    
    # 加载预训练权重的函数
    def load_pretrained_weights(model, weights_path):
        """加载预训练权重到模型"""
        if weights_path is None:
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
        
        print(f"加载预训练权重: {weights_path}")
        return model
    
    # 创建模型函数
    def create_model(pretrained_weights=None):
        """创建模型并加载预训练权重"""
        num_channels = data.shape[1]  # 获取通道数
        
        # 初始化模型
        model = VisionTransformer(
            img_size=config.IMG_SIZE,
            patch_size=config.PATCH_SIZE,
            in_channels=num_channels,
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
    
    # K折交叉验证
    fold_results = []
    
    for fold in range(args.k_folds):
        print(f"\nFold {fold+1}/{args.k_folds}")
        
        # 创建K折数据加载器
        train_transforms = get_train_transforms()
        train_loader, val_loader = create_kfold_loaders(
            data, labels, args.k_folds, fold, args.batch_size, train_transforms
        )
        
        # 创建模型
        model = create_model(pretrained_weights)
        
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
            {'params': backbone_params, 'lr': config.FINETUNE_LR_BACKBONE},
            {'params': head_params, 'lr': config.FINETUNE_LR_HEAD}
        ], weight_decay=config.WEIGHT_DECAY)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.finetune_epochs, 
            eta_min=config.FINETUNE_LR_BACKBONE / 10
        )
        
        # 训练循环
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 训练函数
        def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True):
            """训练一个epoch"""
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            mixup_fn = MixUp(alpha=0.2) if use_mixup else None
            
            for inputs, labels in train_loader:
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

        # 验证函数
        def validate(model, val_loader, criterion, device):
            """验证模型"""
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
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

        # 早停参数
        patience = 15  # 验证指标不再提升的轮数
        early_stop_counter = 0
        early_stop_threshold = 0.0005  # 认为有提升的最小阈值
        best_val_acc = 0
        best_val_loss = float('inf')

        for epoch in range(args.finetune_epochs):
            print(f"\nEpoch {epoch+1}/{args.finetune_epochs}")
            
            # 训练
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc.detach().cpu().item() if torch.is_tensor(train_acc) else float(train_acc))
            
            # 验证
            val_loss, val_acc = validate(
                model, val_loader, criterion, config.DEVICE
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc.detach().cpu().item() if torch.is_tensor(val_acc) else float(val_acc))
            
            # 更新学习率
            scheduler.step()
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 首先检查验证准确率是否已达到高水平
            if val_acc >= 0.99:
                # 当准确率已很高时，转为监控损失值
                if val_loss < best_val_loss - early_stop_threshold:
                    best_val_loss = val_loss
                    best_val_acc = max(best_val_acc, val_acc)  # 同时更新最佳准确率
                    early_stop_counter = 0
                    
                    # 保存最佳模型
                    model_path = os.path.join(args.output_dir, f"best_model_fold{fold+1}.pth")
                    if config.NUM_GPUS > 1:
                        torch.save(model.module.state_dict(), model_path)
                    else:
                        torch.save(model.state_dict(), model_path)
                    print(f"保存最佳模型，验证准确率: {val_acc:.4f}，验证损失: {val_loss:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"验证损失未改善，早停计数: {early_stop_counter}/{patience}")
            else:
                # 准确率尚未达到高水平时，仍以准确率为准
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    
                    # 保存最佳模型
                    model_path = os.path.join(args.output_dir, f"best_model_fold{fold+1}.pth")
                    if config.NUM_GPUS > 1:
                        torch.save(model.module.state_dict(), model_path)
                    else:
                        torch.save(model.state_dict(), model_path)
                    print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"验证准确率未提升，早停计数: {early_stop_counter}/{patience}")
            
            # 早停判断
            if early_stop_counter >= patience:
                print(f"早停触发! {patience}个epoch未见显著改善")
                break
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
        
        print(f"\nFold {fold+1} 结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"宏平均精确率: {metrics['precision_macro']:.4f}")
        print(f"宏平均召回率: {metrics['recall_macro']:.4f}")
        print(f"宏平均F1: {metrics['f1_macro']:.4f}")
        
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
    print("\n所有折的平均结果:")
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision_macro'] for r in fold_results])
    avg_recall = np.mean([r['recall_macro'] for r in fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
    
    print(f"准确率: {avg_accuracy:.4f}")
    print(f"宏平均精确率: {avg_precision:.4f}")
    print(f"宏平均召回率: {avg_recall:.4f}")
    print(f"宏平均F1: {avg_f1:.4f}")
    
    # 保存最佳模型（以最高验证准确率为标准）
    best_fold = max(fold_results, key=lambda x: x['accuracy'])['fold']
    best_model_path = os.path.join(args.output_dir, f"best_model_fold{best_fold}.pth")
    
    # 复制最佳模型
    import shutil
    shutil.copy(best_model_path, args.final_model_path)
    print(f"\n从fold {best_fold}中选择最佳模型作为最终模型。")
    
    print("="*50)

def evaluate(args):
    """模型评估"""
    print("\n" + "="*50)
    print("开始模型评估...")
    
    # 导入评估所需模块
    from models.vit_model import VisionTransformer
    from utils.data_processing import prepare_test_dataset, create_data_loaders
    from utils.metrics import evaluate_model, extract_features
    from utils.visualization import plot_confusion_matrix, plot_classification_metrics, plot_feature_space, visualize_attention
    
    # 检查模型路径
    if not os.path.exists(args.final_model_path):
        raise FileNotFoundError(f"模型文件未找到: {args.final_model_path}")
    
    # 加载测试数据
    test_data, test_labels, _ = prepare_test_dataset(
        args.data_path, 
        use_cache=args.use_cache
    )
    
    # 创建测试数据加载器
    test_loader = create_data_loaders(test_data, test_labels, args.batch_size, shuffle=False)
    
    # 创建模型
    num_channels = test_data.shape[1]  # 获取通道数
    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=num_channels,
        n_classes=config.NUM_CLASSES,
        embed_dim=config.VIT_HIDDEN_SIZE,
        depth=config.VIT_NUM_LAYERS,
        n_heads=config.VIT_NUM_HEADS,
        mlp_ratio=config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE
    )
    
    # 加载预训练权重
    model.load_state_dict(torch.load(args.final_model_path))
    model = model.to(config.DEVICE)
    
    # 评估
    print("评估模型...")
    metrics = evaluate_model(model, test_loader, config.DEVICE)
    
    # 评估目录
    eval_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 保存结果
    results_path = os.path.join(eval_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("评估结果:\n")
        f.write(f"准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"宏平均精确率: {metrics['precision_macro']:.4f}\n")
        f.write(f"宏平均召回率: {metrics['recall_macro']:.4f}\n")
        f.write(f"宏平均F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"加权精确率: {metrics['precision_weighted']:.4f}\n")
        f.write(f"加权召回率: {metrics['recall_weighted']:.4f}\n")
        f.write(f"加权F1: {metrics['f1_weighted']:.4f}\n")
        f.write(config.log_output)
    print(f"\n评估结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"宏平均精确率: {metrics['precision_macro']:.4f}")
    print(f"宏平均召回率: {metrics['recall_macro']:.4f}")
    print(f"宏平均F1: {metrics['f1_macro']:.4f}")
    
    # 绘制混淆矩阵
    print("生成混淆矩阵...")
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        figsize=(12, 10)
    )
    plt.savefig(cm_path)
    plt.close()
    
    # 绘制分类指标
    print("生成分类指标图...")
    metrics_path = os.path.join(eval_dir, "classification_metrics.png")
    plot_classification_metrics(
        metrics['labels'],
        metrics['predictions'],
        figsize=(15, 8)
    )
    plt.savefig(metrics_path)
    plt.close()
    
    # 提取特征并可视化特征空间
    print("提取特征并可视化...")
    features, feature_labels = extract_features(model, test_loader, config.DEVICE)
    
    # 绘制特征空间
    features_path = os.path.join(eval_dir, "feature_space.png")
    plot_feature_space(features, feature_labels, figsize=(12, 10))
    plt.savefig(features_path)
    plt.close()
     # 可视化注意力图
    if args.mode == "evaluate":
        print("可视化注意力图...")
        visualize_samples = min(3, len(test_data))
        for i in range(visualize_samples):
            sample = torch.tensor(test_data[i], dtype=torch.float32)
            for head_idx in range(min(3, config.VIT_NUM_HEADS)):
                attn_path = os.path.join(eval_dir, f"attention_sample{i+1}_head{head_idx+1}.png")
                visualize_attention(
                    model,
                    sample,
                    head_idx=head_idx,
                    layer_idx=-1,
                    save_path=attn_path
                )
    
    print(f"评估完成，结果保存至: {eval_dir}")
    print("="*50)

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 显示基本信息
    print("\n" + "="*50)
    print("基于Vision Transformer的轨道交通传动系统故障诊断")
    print("="*50)
    print(f"运行模式: {args.mode}")
    print(f"设备: {config.DEVICE} ({'GPU' if torch.cuda.is_available() else 'CPU'})")
    print(f"GPU数量: {config.NUM_GPUS}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"输出目录: {args.output_dir}")
    print(f"缓存目录: {config.CACHE_DIR}")
    print(config.log_output)
    print("="*50)
    
    # # # 根据模式运行相应的功能
    # if args.mode in ["preprocess", "all"]:
    #     preprocess(args)
    
    # if args.mode in ["pretrain", "all"]:
    #     pretrain(args)
    
    if args.mode in ["finetune", "all"]:
        finetune(args)
    
    if args.mode in ["evaluate", "all"]:
        evaluate(args)
    
    print("\n" + "="*50)
    print("流程完成!")
    print("="*50)

if __name__ == "__main__":
    main()