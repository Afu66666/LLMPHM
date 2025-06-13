import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import repeat

class PatchEmbed(nn.Module):
    """ 
    图像转换为 patch embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.n_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=self.patch_size, 
                              stride=self.patch_size)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class Attention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D//H)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        self.attn = attn  # 保存注意力权重便于可视化
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """
    多层感知机
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer 模型
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., embed_layer=PatchEmbed, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, 
                                       in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # 类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer 编码器
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        # 归一化和分类头
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # 参数初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # 图像转换为patch序列
        x = self.patch_embed(x)  # (B, N, D)
        
        # 添加类别标记
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer块
        x = self.blocks(x)
        
        # 应用归一化
        x = self.norm(x)
        
        # 使用类别标记进行分类
        return x[:, 0]
    
    def forward(self, x):
        # 获取特征
        x = self.forward_features(x)
        
        # 应用分类头
        x = self.head(x)
        
        return x