import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import PIL
import SimpleITK as sitk
from PIL.Image import Resampling
from skimage.measure import find_contours
from torch.utils.data import Dataset, DataLoader

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class PatchEmbedding(nn.Module):
    """Convert image to patches and embed with conv layers"""
    def __init__(self, in_channels=1, embed_dim=128, patch_size=16):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBlock(in_channels, embed_dim//4, 7, 3),
            ConvBlock(embed_dim//4, embed_dim//2, 3, 1),
            ConvBlock(embed_dim//2, embed_dim, 3, 1),
            nn.MaxPool2d(kernel_size=patch_size//8, stride=patch_size//8)
        )
        
    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, H', W']
        return rearrange(x, 'b c h w -> b (h w) c')  # Flatten to sequence

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = res + x
        
        # MLP
        res = x
        x = self.norm2(x)
        x = res + self.mlp(x)
        return x

class CVT(nn.Module):
    """Convolutional Visual Transformer for cardiac segmentation"""
    def __init__(self, in_channels=1, num_classes=3, embed_dim=128, 
                 num_heads=4, num_layers=4, patch_size=16):
        super().__init__()
        
        # 1. Patch embedding with convs
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        # 2. Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 3. Decoder with transposed convs
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, 4, 2, 1),
            ConvBlock(embed_dim//2, embed_dim//4),
            nn.ConvTranspose2d(embed_dim//4, embed_dim//8, 4, 2, 1),
            ConvBlock(embed_dim//8, embed_dim//16),
            nn.Conv2d(embed_dim//16, num_classes, 1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Encode with conv + transformer
        x = self.patch_embed(x)  # [B, N, embed_dim]
        x = self.transformer(x)
        
        # 2. Reshape back to spatial
        h, w = H // self.patch_embed.proj[-1].stride, W // self.patch_embed.proj[-1].stride
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # 3. Decode with transposed convs
        x = self.decoder(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


# 1. Dataset Class using your existing loading functions
class CardiacDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.transform = transform
        
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_name = patient_dir.name
        
        # Load 2CH ED frame and mask (using your existing functions)
        img = load_echo_image(patient_dir, patient_name, "2CH", "ED")
        mask, _ = sitk_load(patient_dir / f"{patient_name}_2CH_ED_gt.nii.gz")
        
        # Convert to tensors
        img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dim
        mask = torch.LongTensor(mask)  # Class indices 0-3
        
        if self.transform:
            img = self.transform(img)
            
        return img, mask

# 2. Training Setup
def train_model():
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVT(in_channels=1, num_classes=4).to(device)  # 4 classes: background + 3 structures
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.5, 1.0]).to(device))
    
    # Data loading (using your data structure)
    dataset = CardiacDataset(database_nifti_root)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training loop
    for epoch in range(50):
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# 3. Run training
if __name__ == "__main__":
    train_model()




# Example usage
if __name__ == "__main__":
    # 1. Create model
    model = CVT(in_channels=1, num_classes=3)
    
    # 2. Create dummy input (batch of 1, 1 channel, 256x256)
    x = torch.randn(1, 1, 256, 256)
    
    # 3. Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 3, 256, 256]import nump as np




