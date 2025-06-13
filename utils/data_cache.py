import os
import numpy as np
import pickle
import time
import json

def get_cache_path(cache_dir="cache"):
    """
    获取缓存路径并确保目录存在
    """
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def cache_exists(dataset_name, cache_dir="cache"):
    """
    检查缓存文件是否存在
    """
    cache_path = os.path.join(get_cache_path(cache_dir), f"{dataset_name}.pkl")
    return os.path.exists(cache_path)

def save_to_cache(data_dict, dataset_name, cache_dir="cache"):
    """
    保存数据到缓存
    """
    start_time = time.time()
    print(f"正在缓存{dataset_name}数据...", end="")
    
    cache_path = os.path.join(get_cache_path(cache_dir), f"{dataset_name}.pkl")
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f" 完成! 用时: {time.time() - start_time:.2f}秒")
    print(f"缓存保存在: {os.path.abspath(cache_path)}")

def load_from_cache(dataset_name, cache_dir="cache"):
    """
    从缓存加载数据
    """
    start_time = time.time()
    print(f"从缓存加载{dataset_name}数据...", end="")
    
    cache_path = os.path.join(get_cache_path(cache_dir), f"{dataset_name}.pkl")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f" 完成! 用时: {time.time() - start_time:.2f}秒")
    return data

def save_stft_data(train_components, val_components, test_components,
                 train_conditions, val_conditions, test_conditions,
                 train_labels, val_labels, test_labels, 
                 components, stft_params):
    """
    保存划分后的STFT数据
    """
    start_time = time.time()  # 修复：添加计时起点
    
    stft_data = {
        'train_components': train_components,
        'val_components': val_components, 
        'test_components': test_components,
        'train_conditions': train_conditions,
        'val_conditions': val_conditions,
        'test_conditions': test_conditions,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'components': components,
        'stft_params': stft_params,
        'y_train': train_labels,  # 添加兼容旧代码的键
        'y_test': test_labels     # 添加兼容旧代码的键
    }
    
    # 保存到文件
    cache_dir = get_cache_path()
    stft_path = os.path.join(cache_dir, 'stft_data.pkl')
    with open(stft_path, 'wb') as f:
        pickle.dump(stft_data, f)
    
    print(f"已保存STFT数据到: {os.path.abspath(stft_path)}")
    print(f"完成! 用时: {time.time() - start_time:.2f}秒")  # 修复：计算正确的用时

def load_stft_data(cache_dir="cache"):
    """
    加载STFT变换后的数据
    """
    print("\n加载STFT变换后的数据...")
    
    # 检查缓存是否存在
    if not cache_exists("stft_data", cache_dir=cache_dir):
        print("未找到STFT数据缓存!")
        return None
    
    # 加载数据
    stft_cache = load_from_cache("stft_data", cache_dir=cache_dir)
    
    # 打印数据信息
    train_components = stft_cache['train_components']
    test_components = stft_cache['test_components']
    components = stft_cache['components']
    
    print("STFT数据加载成功!")
    print(f"训练样本数: {len(stft_cache['y_train'])}")
    print(f"测试样本数: {len(stft_cache['y_test'])}")
    
    for i, comp_name in enumerate(components):
        print(f"部件 {comp_name}: 训练形状={train_components[i].shape}, 测试形状={test_components[i].shape}")
    
    return stft_cache

def save_intermediate_results(name, data, metadata=None):
    """
    保存中间结果
    """
    # 创建中间结果目录
    intermediate_dir = os.path.join(get_cache_path(), 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # 保存数据
    path = os.path.join(intermediate_dir, f"{name}.npz")
    
    # 转换PyTorch张量为numpy数组(如果需要)
    if hasattr(data, 'detach'):  # 检查是否是PyTorch张量
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    
    np.savez_compressed(path, data=data)
    
    # 保存元数据
    if metadata:
        meta_path = os.path.join(intermediate_dir, f"{name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"中间结果已保存至: {path}")