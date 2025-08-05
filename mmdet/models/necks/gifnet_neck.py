# e2e_mfd/models/necks/gifnet_neck.py
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils.gifnet_utils import TransformerNet

@MODELS.register_module()
class GIFNetNeck(BaseModule):
    def __init__(self, in_chans=256, embed_dim=256, depth=2, num_heads=8, 
                 window_size=8, init_cg=None):
        """
        参数:
            in_chans: 输入通道数 (backbone输出通道)
            embed_dim: Transformer嵌入维度
            depth: Transformer块深度
            num_heads: 注意力头数
            window_size: 窗口注意力窗口大小
        """
        super().__init__(init_cfg)
        # 通道调整层 (匹配Transformer输入)
        self.channel_adjust_in = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        
        # Transformer特征增强
        self.transformer = TransformerNet(
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size
        )
        
        # 输出层 (保持通道数不变)
        self.channel_adjust_out = nn.Conv2d(embed_dim, in_chans, kernel_size=1)

    def forward(self, x):
        """
        输入:
            x: backbone输出的特征图 [B, C, H, W]
        输出:
            增强后的特征图 [B, C, H, W]
        """
        # 调整通道数
        x = self.channel_adjust_in(x)
        
        # Transformer特征增强
        x = self.transformer(x)
        
        # 恢复原始通道数
        return self.channel_adjust_out(x)
