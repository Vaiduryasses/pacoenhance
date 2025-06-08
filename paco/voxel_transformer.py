"""
Sparse Voxel Transformer from PVD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for voxel coordinates"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, coords, resolution):
        """
        Args:
            coords: (N, 3) voxel coordinates
            resolution: voxel grid resolution
        Returns:
            pos_encoding: (N, channels) positional encoding
        """
        # Normalize coordinates to [0, 1]
        coords_norm = coords.float() / (resolution - 1)
        
        # Create frequency bands
        num_freqs = self.channels // 6  # 2 (sin,cos) * 3 (x,y,z)
        freq_bands = torch.pow(2.0, torch.linspace(0, num_freqs-1, num_freqs, device=coords.device))
        
        # Apply sinusoidal encoding
        encodings = []
        for i in range(3):  # x, y, z coordinates
            coord = coords_norm[:, i:i+1]  # (N, 1)
            for freq in freq_bands:
                encodings.append(torch.sin(coord * freq * math.pi))
                encodings.append(torch.cos(coord * freq * math.pi))
        
        pos_encoding = torch.cat(encodings, dim=1)
        
        # Pad if necessary
        if pos_encoding.shape[1] < self.channels:
            padding = torch.zeros(pos_encoding.shape[0], 
                                self.channels - pos_encoding.shape[1], 
                                device=coords.device)
            pos_encoding = torch.cat([pos_encoding, padding], dim=1)
        elif pos_encoding.shape[1] > self.channels:
            pos_encoding = pos_encoding[:, :self.channels]
            
        return pos_encoding


class SparseWindowAttention(nn.Module):
    """Sparse window attention for voxel transformer"""
    
    def __init__(self, dim, window_size, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros((2*window_size-1)**3, num_heads))
        
        # Precompute relative position indices
        self.register_buffer("rel_pos_indices", self._get_rel_pos_indices(window_size))
    
    def _get_rel_pos_indices(self, window_size):
        """Precompute relative position indices for 3D window"""
        coords = torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size), 
            torch.arange(window_size),
            indexing='ij'
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)  # (W^3, 3)
        
        # Compute relative positions
        rel_coords = coords[:, None, :] - coords[None, :, :]  # (W^3, W^3, 3)
        rel_coords += window_size - 1  # Shift to [0, 2*window_size-2]
        
        # Convert to indices
        rel_pos_indices = rel_coords[:, :, 0] * (2*window_size-1)**2 + \
                         rel_coords[:, :, 1] * (2*window_size-1) + \
                         rel_coords[:, :, 2]
        
        return rel_pos_indices
    
    def forward(self, x, coords):
        """
        Args:
            x: (N, dim) voxel features
            coords: (N, 3) voxel coordinates
        Returns:
            out: (N, dim) attended features
        """
        N, dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # Each: (N, num_heads, head_dim)
        
        # Apply scaling
        q = q * self.scale
        
        # Compute attention scores
        attn = torch.einsum('nhd,mhd->nhm', q, k)  # (N, num_heads, N)
        
        # Add relative position bias (simplified for sparse case)
        # In a full implementation, you'd need to compute windows and local attention
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('nhm,mhd->nhd', attn, v)  # (N, num_heads, head_dim)
        out = out.reshape(N, dim)
        
        # Final projection
        out = self.proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with window attention"""
    
    def __init__(self, dim, window_size, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4, dropout)
        
    def forward(self, x, coords):
        """
        Args:
            x: (N, dim) voxel features
            coords: (N, 3) voxel coordinates
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), coords)
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x


class SparseVoxelTransformer(nn.Module):
    """Sparse Voxel Transformer Encoder for global feature extraction"""
    
    def __init__(self, 
                 in_dim=35,
                 dim=256, 
                 depth=8, 
                 num_heads=8, 
                 window_size=3,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # Initial feature projection
        self.feature_projection = nn.Linear(in_dim, dim)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, feats, coords):
        """
        Args:
            feats: (N, 35) voxel features
            coords: (N, 3) voxel coordinates
        Returns:
            global_features: (N, dim) transformed features
        """
        # Project input features
        x = self.feature_projection(feats)  # (N, dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(coords, resolution=64)
        x = x + pos_enc
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, coords)
        
        # Final normalization
        x = self.output_norm(x)
        
        return x