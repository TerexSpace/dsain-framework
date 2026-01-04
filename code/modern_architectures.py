#!/usr/bin/env python3
"""
Modern Neural Network Architectures for Federated Learning
===========================================================

Implements ResNet-18, MobileNetV2, and Vision Transformer (ViT-Tiny)
adapted for CIFAR-10/CIFAR-100 federated learning experiments.

All architectures are modified from standard implementations to work with
32x32 input images and federated learning constraints.

References:
    - ResNet: He et al. (2016) "Deep Residual Learning for Image Recognition"
    - MobileNetV2: Sandler et al. (2018) "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    - ViT: Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# =============================================================================
# ResNet-18 for CIFAR (Modified for 32x32 Images)
# =============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10/100 (32x32 images).

    Modifications from standard ResNet-18:
    - First conv layer: 3x3 kernel, stride=1, no max pooling (CIFAR images are small)
    - Channels: [64, 128, 256, 512] → same as standard
    - Blocks: [2, 2, 2, 2] → ResNet-18 configuration
    - Final avgpool: 4x4 → matches CIFAR spatial dimensions after downsampling

    Parameters:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)

    Shape:
        - Input: (N, 3, 32, 32) where N is batch size
        - Output: (N, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        # Initial convolution - modified for CIFAR (no max pooling)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool for CIFAR (images too small)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x: (N, 64, 32, 32)

        # Residual blocks
        x = self.layer1(x)  # (N, 64, 32, 32)
        x = self.layer2(x)  # (N, 128, 16, 16)
        x = self.layer3(x)  # (N, 256, 8, 8)
        x = self.layer4(x)  # (N, 512, 4, 4)

        # Global pooling and classifier
        x = self.avgpool(x)  # (N, 512, 1, 1)
        x = torch.flatten(x, 1)  # (N, 512)
        x = self.fc(x)  # (N, num_classes)

        return x


# =============================================================================
# MobileNetV2 for CIFAR (Modified for 32x32 Images)
# =============================================================================

class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 bottleneck)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection (linear bottleneck)
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_CIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10/100 (32x32 images).

    Modifications from standard MobileNetV2:
    - First conv layer: stride=1 (instead of stride=2) for small images
    - Reduced channel widths by factor of 0.5 for efficiency in FL
    - Inverted residual blocks with expansion ratios [1, 6, 6, 6, 6, 6, 6]
    - Final conv: 1280 → 640 channels (width multiplier = 0.5)

    Parameters:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        width_mult: Width multiplier for channels (default 0.5 for FL efficiency)

    Shape:
        - Input: (N, 3, 32, 32)
        - Output: (N, num_classes) logits
    """

    def __init__(self, num_classes: int = 10, width_mult: float = 0.5):
        super().__init__()
        input_channel = 32
        last_channel = 640  # Reduced from 1280

        # Inverted residual settings: [expand_ratio, output_channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride=1 instead of 2 for CIFAR
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Apply width multiplier
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)

        # First conv layer
        self.features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Final conv layer
        self.features.append(nn.Conv2d(input_channel, last_channel, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*self.features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# Vision Transformer (ViT-Tiny) for CIFAR
# =============================================================================

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.proj(x)  # (N, embed_dim, 8, 8) for patch_size=4
        x = x.flatten(2)  # (N, embed_dim, 64)
        x = x.transpose(1, 2)  # (N, 64, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_Tiny_CIFAR(nn.Module):
    """
    Vision Transformer (ViT-Tiny) adapted for CIFAR-10/100.

    Architecture:
    - Patch size: 4x4 (32x32 image → 8x8 = 64 patches)
    - Embedding dimension: 192
    - Number of heads: 3
    - Number of layers: 12
    - MLP ratio: 4
    - Parameters: ~5M (efficient for FL)

    Modifications from standard ViT:
    - Smaller patch size (4 instead of 16) for small images
    - Reduced embedding dimension (192 instead of 768)
    - Fewer attention heads (3 instead of 12)
    - Designed for 32x32 input images

    Parameters:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        img_size: Input image size (default 32 for CIFAR)
        patch_size: Patch size (default 4 for CIFAR)
        embed_dim: Embedding dimension (default 192)
        depth: Number of transformer blocks (default 12)
        num_heads: Number of attention heads (default 3)
        mlp_ratio: MLP hidden dimension ratio (default 4.0)
        dropout: Dropout rate (default 0.1)

    Shape:
        - Input: (N, 3, 32, 32)
        - Output: (N, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)

        # Initialize position embedding and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take class token
        x = self.head(cls_token_final)

        return x


# =============================================================================
# Factory Function
# =============================================================================

def get_model(model_name: str, num_classes: int = 10, **kwargs):
    """
    Factory function to create model instances.

    Args:
        model_name: One of ['resnet18', 'mobilenetv2', 'vit_tiny']
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model instance

    Example:
        >>> model = get_model('resnet18', num_classes=10)
        >>> model = get_model('mobilenetv2', num_classes=100, width_mult=0.75)
        >>> model = get_model('vit_tiny', num_classes=10, depth=6)
    """
    models = {
        'resnet18': ResNet18_CIFAR,
        'mobilenetv2': MobileNetV2_CIFAR,
        'vit_tiny': ViT_Tiny_CIFAR
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](num_classes=num_classes, **kwargs)


# =============================================================================
# Model Statistics
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)):
    """
    Print model summary including parameters and FLOPs estimation.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    """
    num_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*70}")
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"Input size: {input_size}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test all models
    print("\n" + "="*70)
    print("TESTING MODERN ARCHITECTURES FOR FEDERATED LEARNING")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)

    # Test ResNet-18
    print("\n1. ResNet-18 for CIFAR-10:")
    model_resnet = ResNet18_CIFAR(num_classes=10).to(device)
    model_summary(model_resnet)
    out_resnet = model_resnet(x)
    print(f"Output shape: {out_resnet.shape}")
    assert out_resnet.shape == (batch_size, 10), f"Expected (4, 10), got {out_resnet.shape}"
    print("[OK] ResNet-18 test passed!")

    # Test MobileNetV2
    print("\n2. MobileNetV2 for CIFAR-100:")
    model_mobile = MobileNetV2_CIFAR(num_classes=100, width_mult=0.5).to(device)
    model_summary(model_mobile)
    out_mobile = model_mobile(x)
    print(f"Output shape: {out_mobile.shape}")
    assert out_mobile.shape == (batch_size, 100), f"Expected (4, 100), got {out_mobile.shape}"
    print("[OK] MobileNetV2 test passed!")

    # Test ViT-Tiny
    print("\n3. ViT-Tiny for CIFAR-10:")
    model_vit = ViT_Tiny_CIFAR(num_classes=10, depth=6).to(device)  # Reduced depth for testing
    model_summary(model_vit)
    out_vit = model_vit(x)
    print(f"Output shape: {out_vit.shape}")
    assert out_vit.shape == (batch_size, 10), f"Expected (4, 10), got {out_vit.shape}"
    print("[OK] ViT-Tiny test passed!")

    # Test factory function
    print("\n4. Testing factory function:")
    for name in ['resnet18', 'mobilenetv2', 'vit_tiny']:
        model = get_model(name, num_classes=10)
        print(f"   {name}: {count_parameters(model):,} parameters")

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70 + "\n")
