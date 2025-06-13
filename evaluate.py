import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import matplotlib.pyplot as plt

from config import config
from models.vit_model import VisionTransformer
from utils.metrics import evaluate_model, extract_features
from utils.visualization import plot_confusion_matrix, plot_classification_metrics, plot_feature_space, visualize_attention

def parse_args():
    parser = argparse.ArgumentParser(description="ViT-based fault diagnosis evaluation")
    parser.add_argument("--processed_data_path", type=str, default=os.path.join(config.DATA_DIR, "processed_data.h5"),
                        help="Path to processed data")
    parser.add_argument("--model_path", type=str, default=os.path.join(config.OUTPUT_DIR, "final_model.pth"),
                        help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default=os.path.join(config.OUTPUT_DIR, "evaluation"),
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--visualize_samples", type=int, default=5,
                        help="Number of samples to visualize attention maps")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载处理后的数据
    print(f"Loading processed data from {args.processed_data_path}...")
    with h5py.File(args.processed_data_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    
    print(f"Loaded processed data: {data.shape}, labels: {labels.shape}")
    
    # 划分测试集 (使用20%的数据作为测试集)
    indices = np.random.permutation(len(data))
    test_size = int(0.2 * len(data))
    test_indices = indices[:test_size]
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(test_data, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True
    )
    
    # 创建模型
    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.NUM_CHANNELS,
        n_classes=config.NUM_CLASSES,
        embed_dim=config.VIT_HIDDEN_SIZE,
        depth=config.VIT_NUM_LAYERS,
        n_heads=config.VIT_NUM_HEADS,
        mlp_ratio=config.VIT_MLP_SIZE / config.VIT_HIDDEN_SIZE
    )
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(config.DEVICE)
    model.eval()
    
    # 评估模型
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, config.DEVICE)
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    
    # 保存结果到文本文件
    with open(os.path.join(args.output_dir, "evaluation_results.txt"), 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n")
    
    # 绘制混淆矩阵
    print("Generating confusion matrix...")
    plt.figure(figsize=(12, 10))
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        figsize=(12, 10)
    )
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()
    
    # 绘制分类指标
    print("Generating classification metrics plot...")
    plt.figure(figsize=(15, 8))
    plot_classification_metrics(
        metrics['labels'],
        metrics['predictions'],
        figsize=(15, 8)
    )
    plt.savefig(os.path.join(args.output_dir, "classification_metrics.png"))
    plt.close()
    
    # 提取特征并可视化特征空间
    print("Extracting features for visualization...")
    features, feature_labels = extract_features(model, test_loader, config.DEVICE)
    
    # 绘制特征空间
    print("Generating feature space visualization...")
    plt.figure(figsize=(12, 10))
    plot_feature_space(features, feature_labels, figsize=(12, 10))
    plt.savefig(os.path.join(args.output_dir, "feature_space.png"))
    plt.close()
    
    # 可视化注意力图
    print("Generating attention visualizations...")
    samples_to_visualize = min(args.visualize_samples, len(test_data))
    
    for i in range(samples_to_visualize):
        input_tensor = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0)
        label = test_labels[i]
        
        for head_idx in range(min(3, config.VIT_NUM_HEADS)):  # 可视化前3个注意力头
            save_path = os.path.join(args.output_dir, f"attention_sample{i+1}_head{head_idx+1}.png")
            visualize_attention(
                model, 
                input_tensor[0], 
                head_idx=head_idx,
                layer_idx=-1,  # 最后一层
                save_path=save_path
            )
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()