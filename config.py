import os
import torch

class Config:
    # 路径配置
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'Datasets')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    CACHE_DIR = os.path.join(ROOT_DIR, 'cache')  # 缓存目录
    
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 数据配置
    COMPONENTS = ["gearbox", "leftaxlebox", "motor", "rightaxlebox"]
    SAMPLE_RATE = 64000  # 每秒采样点数
    FULL_LENGTH = 640000  # 10s数据长度
    TEST_LENGTH = 64000   # 1s测试数据长度
    NUM_CLASSES = 17      # 17个故障类型
    
    # STFT参数
    N_FFT = 512
    HOP_LENGTH = 64
    WIN_LENGTH = 512
    
    # 划分窗口参数
    WINDOW_SIZE = 64000  # 1s
    HOP_SIZE = 8000     # 重叠滑动窗口步长(0.25s)

    # 图像参数
    IMG_SIZE = 224  # ViT输入尺寸
    PATCH_SIZE = 16  # ViT patch大小
    
    # 模型配置
    VIT_HIDDEN_SIZE = 768
    VIT_MLP_SIZE = 3072
    VIT_NUM_HEADS = 12
    VIT_NUM_LAYERS = 4
    DROPOUT_RATE = 0.15
    
    # VIT_HIDDEN_SIZE = 384
    # VIT_MLP_SIZE = 1536
    # VIT_NUM_HEADS = 12
    # VIT_NUM_LAYERS = 6
    # DROPOUT_RATE = 0.15
    
    # VIT_HIDDEN_SIZE = 256
    # VIT_MLP_SIZE = 1024
    # VIT_NUM_HEADS = 8
    # VIT_NUM_LAYERS = 4
    # DROPOUT_RATE = 0.15
    # 训练参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_GPUS = torch.cuda.device_count()
    BATCH = 64
    BATCH_SIZE = BATCH * max(1, NUM_GPUS)  # 每GPU 64个样本
    PRETRAIN_EPOCHS = 100
    FINETUNE_EPOCHS = 50
    PRETRAIN_LR = 1e-4
    FINETUNE_LR_BACKBONE = 1e-5  # 骨干网络微调学习率
    FINETUNE_LR_HEAD = 1e-4      # 分类头学习率
    WARMUP_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    
    # 预训练参数
    MASK_RATIO = 0.75  # MAE掩码比例
    
    # 数据增强参数
    AUG_PROB = 0.5
    AUG_MAGNITUDE = 0.2
    
    # K折交叉验证
    K_FOLDS = 1

    log_output = f"""
--- 本次训练参数配置 ---
缓存目录: {CACHE_DIR}

STFT (短时傅里叶变换) 参数:
- N_FFT (FFT点数): {N_FFT}
- HOP_LENGTH (帧移): {HOP_LENGTH}
- WIN_LENGTH (窗长): {WIN_LENGTH}

数据划分窗口参数:
- WINDOW_SIZE (窗口大小，样本数): {WINDOW_SIZE}
- HOP_SIZE (窗口步长，样本数): {HOP_SIZE}

Vision Transformer (ViT) 模型配置:
- VIT_HIDDEN_SIZE (隐藏层维度): {VIT_HIDDEN_SIZE}
- VIT_MLP_SIZE (MLP层中间维度): {VIT_MLP_SIZE}
- VIT_NUM_HEADS (多头注意力头数): {VIT_NUM_HEADS}
- VIT_NUM_LAYERS (Transformer层数): {VIT_NUM_LAYERS}
- DROPOUT_RATE (Dropout比率): {DROPOUT_RATE}

训练批次参数:
- BATCH: {BATCH}
- 检测到的GPU数量: {NUM_GPUS}
-----------------------
"""
config = Config()