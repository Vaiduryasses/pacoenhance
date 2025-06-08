"""
Proxy Decoder from PVD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Transformer decoder block with cross attention"""
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x, context):
        """
        Args:
            x: (B, num_queries, dim) query embeddings
            context: (B, dim) context features
        Returns:
            x: (B, num_queries, dim) updated queries
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention with context
        x_norm = self.norm2(x)
        context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, num_queries, dim)
        attn_out, _ = self.cross_attn(x_norm, context_expanded, context_expanded)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        
        return x


class ProxyDecoder(nn.Module):
    """Proxy Decoder for predicting plane parameters"""
    
    def __init__(self, dim=256, num_queries=40, num_layers=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, dim))
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Output heads
        self.class_head = nn.Linear(dim, 1)      # plane vs. no-plane
        self.param_head = nn.Linear(dim, 3)      # (r, θ, φ) parameters
        self.inlier_head = nn.Linear(dim, 1)     # inlier distance
        self.conf_head = nn.Linear(dim, 1)       # confidence score
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, context):
        """
        Args:
            context: (B, dim) or (B, 1, dim) context features
        Returns:
            outputs: dict containing plane parameters
        """
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, dim)
        
        B = context.shape[0]
        
        # Initialize queries
        queries = self.query_embed.expand(B, -1, -1)  # (B, num_queries, dim)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries, context.squeeze(1))  # context: (B, dim)
        
        # Final normalization
        queries = self.norm(queries)  # (B, num_queries, dim)
        
        # Generate outputs
        logits = self.class_head(queries).squeeze(-1)  # (B, num_queries)
        params = self.param_head(queries)  # (B, num_queries, 3)
        inlier_dist = self.inlier_head(queries).squeeze(-1)  # (B, num_queries)
        conf = torch.sigmoid(self.conf_head(queries)).squeeze(-1)  # (B, num_queries)
        
        # Convert (r, θ, φ) to normal vector (nx, ny, nz) and distance d
        r = params[..., 0]  # (B, num_queries)
        theta = params[..., 1]  # (B, num_queries)
        phi = params[..., 2]  # (B, num_queries)
        
        # Compute normal vector
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)
        
        # Ensure unit normals
        normals = torch.stack([nx, ny, nz], dim=-1)  # (B, num_queries, 3)
        normals = F.normalize(normals, dim=-1)
        
        # Positive distance
        distances = torch.abs(r)  # (B, num_queries)
        
        # For single batch, squeeze dimensions
        if B == 1:
            logits = logits.squeeze(0)
            normals = normals.squeeze(0)
            distances = distances.squeeze(0)
            inlier_dist = inlier_dist.squeeze(0)
            conf = conf.squeeze(0)
        
        return {
            'logits': logits,
            'normals': normals,
            'distances': distances,
            'inlier_dist': inlier_dist,
            'conf': conf
        }