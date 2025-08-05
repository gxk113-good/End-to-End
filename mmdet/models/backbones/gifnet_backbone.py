# e2e_mfd/models/backbones/gifnet_backbone.py
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils.gifnet_utils import SharedFeatureExtractor

@MODELS.register_module()
class GIFNetBackbone(BaseModule):
    def __init__(self, in_chans=3, init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = SharedFeatureExtractor(in_chans=in_chans)
        # 通道调整卷积，输出256通道特征图
        self.channel_adjust = nn.Conv2d(98, 256, kernel_size=1)
        
    def forward(self, ir_image, vis_image):
        """
        输入:
            ir_image: 红外图像 [B, 3, H, W]
            vis_image: 可见光图像 [B, 3, H, W]
        输出:
            融合特征图 [B, 256, H, W]
        """
        # 拼接红外和可见光图像
        x = torch.cat([ir_image, vis_image], dim=1)
        
        # 提取共享特征
        features = self.backbone(x)
        
        # 调整通道数
        return self.channel_adjust(features)
