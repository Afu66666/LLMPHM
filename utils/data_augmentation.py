import torch
import torch.nn.functional as F
import numpy as np
from config import config

class MixUp:
    """
    MixUp数据增强: 混合两个样本及其标签
    ref: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, target):
        # 获取批次大小
        batch_size = x.size(0)
        
        # 生成混合权重
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # 生成打乱的索引
        index = torch.randperm(batch_size, device=x.device)
        
        # 混合数据
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # 返回混合数据和对应的标签
        return mixed_x, target, target[index], lam

def time_shift(x, max_shift_ratio=0.1):
    """
    时间偏移: 沿时间轴随机移动信号
    """
    shift = int(x.shape[-1] * max_shift_ratio)
    if shift > 0:
        direction = np.random.choice([-1, 1])
        shift_value = np.random.randint(1, shift + 1)
        shift_value = shift_value * direction
        
        # 使用循环移位
        x_shifted = torch.roll(x, shifts=shift_value, dims=-1)
        return x_shifted
    else:
        return x

def frequency_mask(x, max_mask_width=0.1, num_masks=1):
    """
    频率掩码: 遮挡频率域的随机区域
    """
    c, h, w = x.shape
    mask_value = x.mean()
    
    # 计算最大掩码宽度
    max_mask_width = int(h * max_mask_width)
    
    for _ in range(num_masks):
        mask_width = np.random.randint(1, max_mask_width + 1)
        mask_start = np.random.randint(0, h - mask_width + 1)
        
        # 应用掩码到所有通道
        x[:, mask_start:mask_start + mask_width, :] = mask_value
        
    return x

def time_mask(x, max_mask_width=0.1, num_masks=1):
    """
    时间掩码: 遮挡时间域的随机区域
    """
    c, h, w = x.shape
    mask_value = x.mean()
    
    # 计算最大掩码宽度
    max_mask_width = int(w * max_mask_width)
    
    for _ in range(num_masks):
        mask_width = np.random.randint(1, max_mask_width + 1)
        mask_start = np.random.randint(0, w - mask_width + 1)
        
        # 应用掩码到所有通道
        x[:, :, mask_start:mask_start + mask_width] = mask_value
        
    return x

def random_gain(x, min_gain=0.8, max_gain=1.2):
    """
    随机增益: 随机缩放信号强度
    """
    gain = np.random.uniform(min_gain, max_gain)
    return x * gain

class RandomAugment:
    """
    随机应用多种增强方法
    """
    def __init__(self, prob=config.AUG_PROB, magnitude=config.AUG_MAGNITUDE):
        self.prob = prob
        self.magnitude = magnitude
        
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # 从数据增强列表中随机选择
        if np.random.random() < self.prob:
            # 定义增强函数列表和它们对应的参数
            aug_functions = [time_shift, frequency_mask, time_mask, random_gain]
            
            # 随机选择增强函数
            idx = np.random.randint(0, len(aug_functions))
            aug_fn = aug_functions[idx]
            
            # 根据选择的函数设置相应的参数
            if aug_fn == time_shift:
                params = {"max_shift_ratio": self.magnitude}
            elif aug_fn == frequency_mask or aug_fn == time_mask:
                params = {"max_mask_width": self.magnitude, "num_masks": 1}
            elif aug_fn == random_gain:
                params = {"min_gain": 1.0 - self.magnitude, "max_gain": 1.0 + self.magnitude}
            else:
                params = {}
                
            # 应用选择的增强方法
            x = aug_fn(x, **params)
            
        return x

def get_train_transforms():
    """
    获取训练数据变换
    """
    return RandomAugment(prob=config.AUG_PROB, magnitude=config.AUG_MAGNITUDE)