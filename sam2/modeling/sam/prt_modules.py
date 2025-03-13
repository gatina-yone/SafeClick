import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from sam2.modeling.sam.my.GCSA import GCSA
from sam2.modeling.sam.my.CRD import Feature_Reweighting
from sam2.modeling.sam.my_image_encoder import Block, Attention,Block_e1

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ExpertNetwork(nn.Module):
    def __init__(
        self,
        intermediate_dim: int = 1280,  # 中间特征维度
        final_dim: int = 256,  # 最终特征维度
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        初始化专家网络，包含三个transformer层。
        
        参数:
            intermediate_dim (int): 中间特征的维度 (1280)
            final_dim (int): 最终特征的维度 (256)
            ...其他参数保持不变...
        """
        super().__init__()
        
        self.dim_transform = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(final_dim),
            nn.GELU()
        )
        self.E1 = Block_e1(
            dim=final_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            input_size=input_size,
        )
        self.E2 = Block(
            dim=final_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            input_size=input_size,
        )

    def forward(
        self, 
        xi: torch.Tensor,  
        xf: torch.Tensor,  
    ) -> List[torch.Tensor]:
        """
        前向传播函数
        
        参数:
            xi: 中间特征, shape [B, 1280, H, W]
            xf: 最终特征, shape [B, 256, H, W]        
        返回:
            List[torch.Tensor]: 包含三个专家输出的列表，每个输出shape都是[B, 256, H, W]
        """
        if xi.shape[-1] == 128: 
            xi = self.dim_transform(xi)
        xi = xi.permute(0, 2, 3, 1)
        xf = xf.permute(0, 2, 3, 1)
        x1 = self.E1(xi,xf)
        x2 = self.E2(xf)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        
        return [x1, x2]

class Network(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        """
        初始化门控网络
        
        参数:
            embed_dim (int): 特征维度
        """
        super().__init__()

        self.crd = Feature_Reweighting()

        self.norm = LayerNorm2d(embed_dim)
        self.activation = nn.LeakyReLU(inplace=True)
        
        self.conv0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.conv0.weight, 0)
        nn.init.constant_(self.conv0.bias, 0)
        
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)

    def forward(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        x,_ = self.crd(expert_outputs[0],expert_outputs[1])
        x = self.conv0(x) + expert_outputs[2]  
        x = self.norm(x)
        x = self.activation(x)
        
        return x