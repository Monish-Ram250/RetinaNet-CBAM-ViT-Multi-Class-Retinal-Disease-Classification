import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models



class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return enhanced



class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))



class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x



class DualBackboneCBAMViT(nn.Module):
    def __init__(self, num_classes, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()

        # EfficientNet-B0
        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.efficientnet = eff.features

        self.eff_proj = nn.Sequential(
            nn.Conv2d(1280, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # ResNet50
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.res_proj = nn.Sequential(
            nn.Conv2d(2048, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Fusion + CBAM
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Attention Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        eff_feat = self.efficientnet(x)          # (B, 1280, 7, 7)
        eff_feat = self.eff_proj(eff_feat)        # (B, embed_dim, 7, 7)

        res_feat = self.resnet(x)                 # (B, 2048, 7, 7)
        res_feat = self.res_proj(res_feat)        # (B, embed_dim, 7, 7)

        fused = torch.cat([eff_feat, res_feat], dim=1)  # (B, embed_dim*2, 7, 7)

        fused = self.fusion_proj(fused)           # (B, embed_dim, 7, 7)
        fused = self.cbam(fused)

        B, C, H, W = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        transformer_out = self.transformer(tokens)

        attn_scores = self.attention_pool(transformer_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled = torch.sum(attn_weights * transformer_out, dim=1)

        output = self.classifier(pooled)

        return output
