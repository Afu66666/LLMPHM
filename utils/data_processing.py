import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import librosa
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2  # 优化：使用OpenCV的图像处理

from config import Config
from utils.data_cache import cache_exists, save_to_cache, load_from_cache, save_intermediate_results
from utils.data_loader import DataLoader as CustomDataLoader

def sliding_window(data, window_size=Config.WINDOW_SIZE, hop_size=Config.HOP_SIZE):
    """
    使用滑动窗口切分时序数据
    
    Args:
        data: 形状为(n_channels, n_timesteps)的数据
        window_size: 窗口大小
        hop_size: 窗口步长
    
    Returns:
        windows: 形状为(n_windows, n_channels, window_size)的窗口数据
    """
    n_channels, n_timesteps = data.shape
    n_windows = (n_timesteps - window_size) // hop_size + 1
    windows = np.zeros((n_windows, n_channels, window_size))
    
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        windows[i] = data[:, start:end]
    
    return windows

def compute_stft(signal, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH):
    """
    计算STFT
    
    Args:
        signal: 形状为(n_timesteps,)的信号
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
    
    Returns:
        magnitude: STFT幅值谱
    """
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude = np.abs(stft)
    # 对数变换压缩动态范围
    magnitude = np.log1p(magnitude)
    return magnitude

def signal_to_image(signal_window, target_size=Config.IMG_SIZE):
    """
    将信号窗口转换为时频图像
    
    Args:
        signal_window: 形状为(n_channels, window_size)的信号窗口
        target_size: 目标图像大小
    
    Returns:
        image: 形状为(n_channels, target_size, target_size)的时频图像
    """
    n_channels, window_size = signal_window.shape
    images = np.zeros((n_channels, Config.N_FFT // 2 + 1, window_size // Config.HOP_LENGTH + 1))
    
    for c in range(n_channels):
        # 计算STFT
        mag = compute_stft(signal_window[c])
        images[c] = mag
    
    # 标准化每个通道
    for c in range(n_channels):
        mean = np.mean(images[c])
        std = np.std(images[c])
        if std > 0:
            images[c] = (images[c] - mean) / std
    
    # 优化：使用OpenCV的resize函数代替手动实现的双线性插值
    resized_images = np.zeros((n_channels, target_size, target_size))
    for c in range(n_channels):
        resized_images[c] = cv2.resize(images[c], (target_size, target_size), 
                                      interpolation=cv2.INTER_LINEAR)
    
    return resized_images

def prepare_dataset(data_dir, use_cache=True):
    """
    准备数据集：加载原始数据，切分窗口，计算STFT
    
    Args:
        data_dir: 原始数据目录路径
        use_cache: 是否使用缓存
    
    Returns:
        processed_data: 处理后的数据
        processed_labels: 处理后的标签
        processed_conditions: 处理后的条件
    """
    dataset_name = "processed_data_stft"
    
    # 检查是否有缓存
    if use_cache and cache_exists(dataset_name, cache_dir=Config.CACHE_DIR):
        print(f"从缓存加载处理后的数据...")
        cache_data = load_from_cache(dataset_name, cache_dir=Config.CACHE_DIR)
        return cache_data['data'], cache_data['labels'], cache_data['conditions']
    
    print(f"处理数据目录: {data_dir}...")
    
    # 使用DataLoader加载数据
    data_loader = CustomDataLoader(data_dir)
    
    # 获取训练数据
    train_components, train_labels, train_conditions = data_loader.get_processed_dataset(dataset="Training")
    if train_components is None:
        raise ValueError("无法获取训练数据，请检查数据路径和格式")
    
    # 处理后的数据和标签列表
    processed_data = []
    processed_labels = []
    processed_conditions = []
    
    # 合并所有组件数据，按样本处理
    num_samples = len(train_labels)
    
    for i in tqdm(range(num_samples), desc="处理样本"):
        # 合并所有组件的通道
        all_channels = []
        
        # 获取每个组件的数据并合并通道
        for component in Config.COMPONENTS:
            component_data = train_components[component][i]
            n_channels = component_data.shape[1]
            
            # 将每个通道添加到all_channels
            for c in range(n_channels):
                all_channels.append(component_data[:, c])
        
        # 将所有通道转换为numpy数组 (n_channels, n_timesteps)
        all_channels = np.array(all_channels)
        
        # 切分窗口
        windows = sliding_window(all_channels, Config.WINDOW_SIZE, Config.HOP_SIZE)
        n_windows = windows.shape[0]
        
        for w in range(n_windows):
            # 转换为时频图像
            img = signal_to_image(windows[w], Config.IMG_SIZE)
            processed_data.append(img)
            processed_labels.append(train_labels[i]) 
            processed_conditions.append(train_conditions[i])
            
            # 保存中间结果示例(每个样本的第一个窗口)
            if w == 0 and i < 5:  # 只保存前5个样本
                save_intermediate_results(
                    f"sample_{i}_window_0_stft", 
                    img,
                    {'label': int(train_labels[i] - 1), 'sample_idx': i, 'window_idx': 0}
                )
    
    # 转换为numpy数组
    processed_data = np.array(processed_data)
    processed_labels = np.array(processed_labels)
    processed_conditions = np.array(processed_conditions)
    
    print(f"处理后的数据形状: {processed_data.shape}")
    print(f"处理后的标签形状: {processed_labels.shape}")
    print(f"处理后的条件形状: {processed_conditions.shape}")
    
    # 保存到缓存
    if use_cache:
        cache_data = {
            'data': processed_data,
            'labels': processed_labels,
            'conditions': processed_conditions,
            'stft_params': {
                'n_fft': Config.N_FFT,
                'hop_length': Config.HOP_LENGTH,
                'window_size': Config.WINDOW_SIZE,
                'hop_size': Config.HOP_SIZE
            }
        }
        save_to_cache(cache_data, dataset_name, cache_dir=Config.CACHE_DIR)
    
    return processed_data, processed_labels, processed_conditions

class FaultDataset(Dataset):
    """故障诊断数据集"""
    
    def __init__(self, data, labels, transform=None):
        """
        初始化数据集
        
        Args:
            data: 形状为(n_samples, n_channels, height, width)的数据
            labels: 形状为(n_samples,)的标签
            transform: 可选的数据变换
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # 转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        # 应用变换
        if self.transform:
            x = self.transform(x)
        
        return x, y

def create_data_loaders(data, labels, batch_size=Config.BATCH_SIZE, transform=None, shuffle=True, num_workers=None):
    """
    创建DataLoader
    
    Args:
        data: 形状为(n_samples, n_channels, height, width)的数据
        labels: 形状为(n_samples,)的标签
        batch_size: 批次大小
        transform: 可选的数据变换
        shuffle: 是否打乱数据
        num_workers: 工作进程数，默认为None，会使用min(8, os.cpu_count() or 1)
    
    Returns:
        data_loader: DataLoader对象
    """
    dataset = FaultDataset(data, labels, transform)
    
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)  # 修复：添加or 1防止os.cpu_count()返回None
        
    data_loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return data_loader

def create_kfold_loaders(data, labels, k=Config.K_FOLDS, fold_idx=0, batch_size=Config.BATCH_SIZE, transform=None):
    """
    创建K折交叉验证的训练和验证DataLoader，支持k=1的单次分层抽样
    
    Args:
        data: 形状为(n_samples, n_channels, height, width)的数据
        labels: 形状为(n_samples,)的标签
        k: 折数，当k=1时执行单次分层抽样
        fold_idx: 当前使用的折索引
        batch_size: 批次大小
        transform: 可选的数据变换
    
    Returns:
        train_loader: 训练DataLoader
        val_loader: 验证DataLoader
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    if k == 1:
        # 当k=1时，执行单次80/20分层抽样
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)), 
            test_size=0.2,
            stratify=labels, 
            random_state=42
        )
    else:
        # 使用StratifiedKFold确保每折类别分布一致
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        folds = list(skf.split(np.arange(len(labels)), labels))
        train_idx, val_idx = folds[fold_idx]
    
    train_data, train_labels = data[train_idx], labels[train_idx]
    val_data, val_labels = data[val_idx], labels[val_idx]
    
    train_loader = create_data_loaders(train_data, train_labels, batch_size, transform, shuffle=True)
    val_loader = create_data_loaders(val_data, val_labels, batch_size, None, shuffle=False)
    
    return train_loader, val_loader

def prepare_test_dataset(data_dir, use_cache=True):
    """
    准备测试数据集
    
    Args:
        data_dir: 测试数据目录
        use_cache: 是否使用缓存
    
    Returns:
        test_data: 测试数据
        test_labels: 测试标签
        test_conditions: 测试条件
    """
    dataset_name = "test_data_stft"
    
    # 检查是否有缓存
    if use_cache and cache_exists(dataset_name, cache_dir=Config.CACHE_DIR):
        print(f"从缓存加载测试数据...")
        cache_data = load_from_cache(dataset_name, cache_dir=Config.CACHE_DIR)
        return cache_data['data'], cache_data['labels'], cache_data['conditions']
    
    print(f"处理测试数据目录: {data_dir}...")
    
    # 使用DataLoader加载测试数据
    data_loader = CustomDataLoader(data_dir)
    
    # 获取测试数据
    test_components, test_labels, test_conditions = data_loader.get_processed_dataset(dataset="Test")
    if test_components is None:
        raise ValueError("无法获取测试数据，请检查数据路径和格式")
    
    # 处理后的数据和标签列表
    processed_data = []
    processed_labels = []
    processed_conditions = []
    
    # 合并所有组件数据，按样本处理
    num_samples = len(test_labels)
    
    for i in tqdm(range(num_samples), desc="处理测试样本"):
        # 合并所有组件的通道
        all_channels = []
        
        # 获取每个组件的数据并合并通道
        for component in Config.COMPONENTS:
            component_data = test_components[component][i]
            n_channels = component_data.shape[1]
            
            # 将每个通道添加到all_channels
            for c in range(n_channels):
                all_channels.append(component_data[:, c])
        
        # 将所有通道转换为numpy数组 (n_channels, n_timesteps)
        all_channels = np.array(all_channels)
        
        # 测试数据窗口长度为1s
        window_size = Config.TEST_LENGTH
        hop_size = window_size  # 无重叠窗口
        
        # 切分窗口
        windows = sliding_window(all_channels, window_size, hop_size)
        n_windows = windows.shape[0]
        
        for w in range(n_windows):
            # 转换为时频图像
            img = signal_to_image(windows[w], Config.IMG_SIZE)
            processed_data.append(img)
            processed_labels.append(test_labels[i]) 
            processed_conditions.append(test_conditions[i])
    
    # 转换为numpy数组
    processed_data = np.array(processed_data)
    processed_labels = np.array(processed_labels)
    processed_conditions = np.array(processed_conditions)
    
    print(f"处理后的测试数据形状: {processed_data.shape}")
    print(f"处理后的测试标签形状: {processed_labels.shape}")
    
    # 保存到缓存
    if use_cache:
        cache_data = {
            'data': processed_data,
            'labels': processed_labels,
            'conditions': processed_conditions,
            'stft_params': {
                'n_fft': Config.N_FFT,
                'hop_length': Config.HOP_LENGTH,
                'window_size': window_size,
                'hop_size': hop_size
            }
        }
        save_to_cache(cache_data, dataset_name, cache_dir=Config.CACHE_DIR)
    
    return processed_data, processed_labels, processed_conditions