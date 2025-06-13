import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from models.vit_model import PatchEmbed, Block
from config import config

class ProjectionHead(nn.Module):
    """用于对比学习的投影头"""
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class MaskedAutoencoderViT(nn.Module):
    """ 
    基于Vision Transformer的掩码自编码器(MAE)
    实现自监督学习的预训练任务，集成对比学习
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3,
        embed_dim=768, 
        depth=12,
        n_heads=12,
        decoder_embed_dim=512, 
        decoder_depth=8,
        decoder_n_heads=16,
        mlp_ratio=4., 
        norm_layer=nn.LayerNorm, 
        norm_pix_loss=False,
        mask_ratio=0.75,
        # 对比学习相关参数
        use_contrastive=True,
        contrastive_dim=128,
        temperature=0.5
    ):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE编码器
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE解码器
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_n_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # 对比学习组件
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        
        if use_contrastive:
            # 对比学习投影头
            self.projection_head = ProjectionHead(
                in_dim=embed_dim,
                hidden_dim=embed_dim,
                out_dim=contrastive_dim
            )
        # --------------------------------------------------------------------------
        
        # 初始化权重
        self.initialize_weights()
        
        # 损失规范化
        self.norm_pix_loss = norm_pix_loss
        
        # 保存掩码比例
        self.mask_ratio = mask_ratio
        
    def initialize_weights(self):
        # 初始化位置嵌入
        pos_embed = self._get_pos_embed(self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)
        
        decoder_pos_embed = self._get_pos_embed(self.decoder_pos_embed.shape[-1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)
        
        # 初始化其他参数
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _get_pos_embed(self, embed_dim):
        # 简单的位置嵌入初始化
        grid_size = int(self.patch_embed.n_patches**0.5)
        pos_embed = torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        return pos_embed

    def random_masking(self, x, mask_ratio):
        """
        执行随机掩码操作
        x: [B, N, D], 序列
        mask_ratio: 掩码比例
        """
        B, N, D = x.shape  # batch, length, dim
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # 噪声 [B, N]
        
        # 保持噪声较小的 len_keep 个位置 (排除 cls token)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前 len_keep 个位置
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成掩码: 1表示缺少的位置, 0表示存在的位置
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # 恢复顺序 (根据 ids_restore 排序)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        MAE编码器: 对输入进行掩码后编码可见部分
        """
        # 嵌入patches
        x = self.patch_embed(x)
        
        # 添加位置嵌入
        x = x + self.pos_embed[:, 1:, :]
        
        # 应用掩码: 保留 (1-mask_ratio) 的patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # 添加 cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 应用Transformer块
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        MAE解码器: 重建完整图像
        """
        # 嵌入tokens
        x = self.decoder_embed(x)
        
        # 创建一个与完整序列长度相同的mask_token序列
        B = x.shape[0]
        n_patches = ids_restore.shape[1]
        n_visible = x.shape[1] - 1
        
        # 创建足够的mask tokens (用于被遮蔽的位置)
        mask_tokens = self.mask_token.repeat(B, n_patches - n_visible, 1)
        
        # 将可见块的输出和mask_token拼接
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 去掉cls_token，拼接
        
        # 根据ids_restore重排序，恢复原始patch顺序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # 添加cls_token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # 添加位置嵌入
        x = x + self.decoder_pos_embed
        
        # 应用解码器Transformer块
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 预测像素
        x = self.decoder_pred(x)
        
        # 移除cls_token
        x = x[:, 1:, :]
        
        return x
    
    def forward_contrastive(self, latent):
        """
        提取用于对比学习的特征
        """
        # 使用CLS token作为全局表示
        cls_token = latent[:, 0]
        return self.projection_head(cls_token)

    def forward(self, imgs, imgs2=None, mask_ratio=None):
        """
        前向传播: 支持MAE和对比学习
        
        Args:
            imgs: 第一组输入图像
            imgs2: 第二组输入图像（用于对比学习）
            mask_ratio: 掩码比例，如果为None则使用默认值
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # 编码第一组图像
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # 预测像素
        pred = self.forward_decoder(latent, ids_restore)
        
        # 如果开启了对比学习并且提供了第二组图像
        if self.use_contrastive and imgs2 is not None:
            # 编码第二组图像（不需要解码）
            latent2, _, _ = self.forward_encoder(imgs2, mask_ratio)
            
            # 获取对比学习投影
            proj1 = self.forward_contrastive(latent)
            proj2 = self.forward_contrastive(latent2)
            
            return pred, mask, latent, proj1, proj2
        
        return pred, mask, latent

    def patchify(self, imgs):
        """
        将图像转化为patches
        """
        p = self.patch_embed.patch_size
        if isinstance(p, tuple):
            p = p[0]  # 假设是正方形
            
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        
        # 重塑为patches
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, H // p * W // p, C * p * p)
        return x

    def unpatchify(self, x, channels=None):
        """
        将patches重塑为图像
        """
        p = self.patch_embed.patch_size
        if isinstance(p, tuple):
            p = p[0]  # 假设是正方形
            
        # 获取解码器输出形状
        B, N, L = x.shape
        H = W = int(N ** 0.5)
        C = L // (p*p) if channels is None else channels
        
        # 重塑为图像
        x = x.reshape(B, H, W, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H * p, W * p)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        计算MAE重建损失: 仅在掩码区域计算重建MSE
        """
        # 将图像转化为patches
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            # 按照每个patch进行归一化
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # 计算重建损失 (仅在掩码区域)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], N是patch数量
        
        # 仅计算掩码部分的损失
        loss = (loss * mask).sum() / mask.sum()  # 平均每个掩码patch的损失
        
        return loss
    
    def contrastive_loss(self, proj1, proj2, temperature=None):
        """
        计算对比学习损失 (NT-Xent)
        
        Args:
            proj1: 第一组图像的投影表示
            proj2: 第二组图像的投影表示
            temperature: 温度参数，如果为None则使用默认值
        """
        if temperature is None:
            temperature = self.temperature
            
        # 将两组特征拼接起来
        batch_size = proj1.size(0)
        representations = torch.cat([proj1, proj2], dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                               representations.unsqueeze(0), 
                                               dim=2)
        
        # 掩码排除自身相似度
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # 掩码排除同一增强样本间相似度
        mask = ~torch.eye(batch_size * 2, dtype=bool, device=proj1.device)
        negatives = similarity_matrix[mask].view(batch_size * 2, -1)
        
        # 计算损失
        positives = positives / temperature
        negatives = negatives / temperature
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=proj1.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def compute_total_loss(self, imgs, imgs2, alpha=0.5):
        """
        计算组合损失: MAE重建损失和对比学习损失
        
        Args:
            imgs: 第一组输入图像
            imgs2: 第二组输入图像（用于对比学习）
            alpha: 控制两种损失权重的系数, alpha*MAE_loss + (1-alpha)*contrastive_loss
        """
        # 前向传播获取所有输出
        pred, mask, latent, proj1, proj2 = self.forward(imgs, imgs2)
        
        # 计算MAE重建损失
        mae_loss = self.forward_loss(imgs, pred, mask)
        
        # 计算对比学习损失
        con_loss = self.contrastive_loss(proj1, proj2)
        
        # 计算总损失
        total_loss = alpha * mae_loss + (1 - alpha) * con_loss
        
        return total_loss, mae_loss, con_loss, pred, mask