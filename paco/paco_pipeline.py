from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils

from . import MODELS
from .transformer_utils import (
    LayerScale, MLP, Attention, DeformableLocalAttention,
    DeformableLocalCrossAttention, DynamicGraphAttention,
    ImprovedDeformableLocalGraphAttention, CrossAttention,
    knn_point, index_points
)

import torch
import torch.nn as nn
from .paco_pvd_pipeline import PaCoVoxelPipeline, PaCoVoxelLoss, Fold, SimpleRebuildFCLayer
from . import MODELS


class SelfAttnBlockAPI(nn.Module):
    r"""
    Self Attention Block API

    This block supports different configurations:
      1. Norm Encoder Block (block_style = 'attn')
      2. Concatenation Fused Encoder Block (block_style = 'attn-deform', combine_style = 'concat')
      3. Three-layer Fused Encoder Block (block_style = 'attn-deform', combine_style = 'onebyone')
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            block_style='attn-deform',
            combine_style='concat',
            k=10,
            n_group=2
    ):
        super().__init__()
        self.combine_style = combine_style
        # Ensure the combine style is valid
        assert combine_style in ['concat', 'onebyone'], (
            f'got unexpect combine_style {combine_style} for local and global attn'
        )
        self.norm1 = norm_layer(dim)
        # Apply LayerScale if init_values is provided, else identity
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Process block style tokens (e.g. 'attn' or 'attn-deform')
        block_tokens = block_style.split('-')
        assert 0 < len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect block_token {block_token} for Block component'
            )
            if block_token == 'attn':
                self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
                
        # If both global and local attentions are used, set up merging
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        """
        Forward pass for the SelfAttnBlockAPI

        Args:
            x: Input feature tensor
            pos: Positional encoding tensor
            idx: (Optional) Index tensor for local attention computations

        Returns:
            Output tensor after self-attention and MLP operations
        """
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # Combine features via concatenation and linear mapping
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))
        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # If only one attention branch is used, add it directly
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()
        # Apply MLP block after attention
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlockAPI(nn.Module):
    r"""
    Cross Attention Block API

    This block supports multiple configurations for self-attention and cross-attention.
    Examples of supported designs include:
      1. Norm Decoder Block (self_attn_block_style = 'attn', cross_attn_block_style = 'attn')
      2. Concatenation Fused Decoder Block (self_attn_block_style = 'attn-deform', combine_style = 'concat')
      3. Three-layer Fused Decoder Block (self_attn_block_style = 'attn-deform', combine_style = 'onebyone')
      4. Custom designs mixing plain and deformable attention or graph convolution
    """

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Self-attention branch initialization
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], (
            f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
        )

        self_attn_block_tokens = self_attn_block_style.split('-')
        assert 0 < len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            )
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Cross-attention branch initialization
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], (
            f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        )

        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert 0 < len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            )
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(
                    dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        """
        Forward pass for the CrossAttnBlockAPI

        Args:
            q: Query tensor
            v: Value tensor
            q_pos: Positional encoding for query
            v_pos: Positional encoding for value
            self_attn_idx: (Optional) Index tensor for self-attention neighborhood
            cross_attn_idx: (Optional) Index tensor for cross-attention neighborhood
            denoise_length: (Optional) Length for denoising tokens

        Returns:
            Updated query tensor after cross-attention and MLP operations
        """
        # Determine mask if denoising is applied
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self-attention branch
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(
                        norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length
                    )
                    feature_list.append(local_attn_feat)
                # Combine self-attention features
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(
                    self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                ))
        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(
                    norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length
                )
                feature_list.append(local_attn_feat)
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # Cross-attention branch
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(
                        q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                    )
                    feature_list.append(local_attn_feat)
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(
                    self.local_cross_attn(
                        q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                    )
                ))
        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(
                    q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                )
                feature_list.append(local_attn_feat)
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder without hierarchical structure

    This encoder applies a sequence of SelfAttnBlockAPI blocks.
    """

    def __init__(
            self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2
    ):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SelfAttnBlockAPI(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    block_style=block_style_list[i],
                    combine_style=combine_style,
                    k=k,
                    n_group=n_group
                )
            )

    def forward(self, x, pos):
        # If positional information is provided, compute k-NN indices
        if pos is not None:
            idx = knn_point(self.k, pos, pos)
        else:
            idx = None
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder without hierarchical structure

    This decoder applies a sequence of CrossAttnBlockAPI blocks.
    """

    def __init__(
            self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                CrossAttnBlockAPI(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    self_attn_block_style=self_attn_block_style_list[i],
                    self_attn_combine_style=self_attn_combine_style,
                    cross_attn_block_style=cross_attn_block_style_list[i],
                    cross_attn_combine_style=cross_attn_combine_style,
                    k=k,
                    n_group=n_group
                )
            )

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        if q_pos is None:
            cross_attn_idx = None
        else:
            cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(
                q, v, q_pos, v_pos,
                self_attn_idx=self_attn_idx,
                cross_attn_idx=cross_attn_idx,
                denoise_length=denoise_length
            )
        return q


class PointTransformerEncoder(nn.Module):
    """
    Vision Transformer for point cloud encoder

    A PyTorch implementation inspired by:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group
        )
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos=None):
        x = self.blocks(x, pos)
        return x


class PointTransformerDecoder(nn.Module):
    """
    Vision Transformer for point cloud decoder

    A PyTorch implementation inspired by:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list,
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list,
            cross_attn_combine_style=cross_attn_combine_style,
            k=k,
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q


# Entry wrappers to build encoder/decoder from configuration
class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class DGCNN_Grouper(nn.Module):
    """
    Dynamic Graph CNN Grouper

    Groups points using DGCNN and applies several convolutional layers to extract features.
    """

    def __init__(self, k=16):
        super().__init__()
        # K must be 16
        self.k = k
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, normal, plane_idx, num_group):
        """
        Farthest point sampling downsample

        Args:
            coor: Coordinates tensor (B, C, N)
            x: Feature tensor
            normal: Normal vectors tensor
            plane_idx: Plane index tensor
            num_group: Number of groups

        Returns:
            new_coor, new_normal, new_x, new_plane_idx after sampling
        """
        xyz = coor.transpose(1, 2).contiguous()  # B, N, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        combined_x = torch.cat([coor, normal, plane_idx, x], dim=1)
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)
        new_coor = new_combined_x[:, :3, :]
        new_normal = new_combined_x[:, 3:6, :]
        new_plane_idx = new_combined_x[:, 6, :].unsqueeze(1)
        new_x = new_combined_x[:, 7:, :]
        return new_coor, new_normal, new_x, new_plane_idx

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        """
        Compute graph features using k-NN

        Args:
            coor_q: Query coordinates tensor
            x_q: Query feature tensor
            coor_k: Key coordinates tensor
            x_k: Key feature tensor

        Returns:
            Graph feature tensor
        """
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(
                k,
                coor_k.transpose(-1, -2).contiguous(),
                coor_q.transpose(-1, -2).contiguous()
            )
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        """
        Forward pass for grouping

        Args:
            x: Input tensor with shape (B, N, 7) where 7 corresponds to x,y,z,nx,ny,nz,plane_id
            num: List with grouping parameters

        Returns:
            coor, f, normal, plane_idx after grouping
        """
        assert x.shape[-1] == 7
        batch_size, num_points, _ = x.size()
        x = x.transpose(-1, -2).contiguous()
        coor, normal, plane_idx = x[:, :3, :], x[:, 3:6, :], x[:, -1, :].unsqueeze(1)
        f = self.input_trans(coor)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, normal_q, f_q, plane_idx_q = self.fps_downsample(coor, f, normal, plane_idx, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor, normal, plane_idx = coor_q, normal_q, plane_idx_q
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, normal_q, f_q, plane_idx_q = self.fps_downsample(coor, f, normal, plane_idx, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor, normal, plane_idx = coor_q, normal_q, plane_idx_q
        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()
        normal = normal.transpose(-1, -2).contiguous()
        plane_idx = plane_idx.transpose(-1, -2).contiguous()
        return coor, f, normal, plane_idx


class Encoder(nn.Module):
    """
    Encoder module for extracting global features from point groups

    Processes point groups with sequential convolutions.
    """

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        Forward pass for the Encoder

        Args:
            point_groups: Tensor with shape (B, G, N, 3)

        Returns:
            feature_global: Tensor with shape (B, G, C)
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class SimpleEncoder(nn.Module):
    """
    Simple Encoder for point clouds using farthest point sampling and grouping
    """

    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.num_features = embed_dims

    @staticmethod
    def fps(data, number):
        """
        Farthest point sampling

        Args:
            data: Tensor with shape (B, N, 7)
            number: Number of points to sample

        Returns:
            Sampled data tensor
        """
        fps_idx = pointnet2_utils.furthest_point_sample(data[:, :, :3], number)
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).contiguous()
        return fps_data

    def forward(self, xyz, n_group):
        """
        Forward pass for the SimpleEncoder

        Args:
            xyz: Input point cloud tensor (B, N, 7)
            n_group: Number of groups to sample

        Returns:
            center, features, normal, plane_idx tensors
        """
        if isinstance(n_group, list):
            n_group = n_group[-1]
        assert xyz.shape[-1] == 7
        batch_size, num_points, _ = xyz.shape
        coor = xyz[:, :, :3]
        xyz = self.fps(xyz, n_group)
        center, normal, plane_idx = xyz[:, :3, :], xyz[:, 3:6, :], xyz[:, -1, :].unsqueeze(1)
        idx = knn_point(self.group_size, coor, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = coor.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
        features = self.embedding(neighborhood)
        return center, features, normal, plane_idx


class Fold(nn.Module):
    """
    Folding module to generate point clouds from latent features

    This module uses two folding layers to reconstruct a point cloud.
    """

    def __init__(self, in_channel, step, hidden_dim=512, freedom=3):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.freedom = freedom

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, self.freedom, 1)
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + self.freedom, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, self.freedom, 1)
        )

    def forward(self, x):
        """
        Forward pass for the folding module

        Args:
            x: Input latent feature tensor

        Returns:
            Reconstructed point cloud offsets
        """
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2


class SimpleRebuildFCLayer(nn.Module):
    """
    Simple Fully Connected Layer for Rebuilding Point Clouds

    Applies an MLP on concatenated global and token features.
    """

    def __init__(self, input_dims, step, hidden_dim=512, freedom=3):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.freedom = freedom
        self.layer = MLP(self.input_dims, hidden_dim, step * self.freedom)

    def forward(self, rec_feature):
        """
        Forward pass for rebuilding

        Args:
            rec_feature: Reconstruction feature tensor with shape (B, N, C)

        Returns:
            Rebuilt point cloud tensor with shape (B, N, step, freedom)
        """
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, self.freedom)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder = config.encoder
        decoder = config.decoder
        self.num_centers = getattr(config, 'num_centers', [512, 128])
        self.num_planes = getattr(config, 'num_planes', 20)
        self.group_k = getattr(config, 'group_k', 32)
        self.query_ranking = getattr(config, 'query_ranking', False)
        self.encoder_type = config.encoder_type
        self.query_type = config.query_type
        in_chans = 3
        self.num_queries = config.num_queries
        query_num = config.num_queries
        global_feature_dim = config.global_feature_dim
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=self.group_k)
            self.plane_mlp = nn.Sequential(
                nn.Linear(encoder.embed_dim * 2, encoder.embed_dim),
                nn.GELU()
            )
        else:
            self.grouper = SimpleEncoder(k=self.group_k, num_planes=self.num_planes)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder.embed_dim)
        )
        self.plane_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder.embed_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder.embed_dim)
        )
        self.encoder = PointTransformerEncoderEntry(encoder)
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        if self.query_type == 'dynamic':
            self.plane_pred_coarse = nn.Sequential(
                nn.Linear(global_feature_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, 3 * query_num))
            self.mlp_query = nn.Sequential(
                nn.Linear(global_feature_dim + 3, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Linear(1024, decoder.embed_dim))
            self.plane_pred = nn.Sequential(
                nn.Linear(decoder.embed_dim, decoder.embed_dim // 2),
                nn.GELU(),
                nn.Linear(decoder.embed_dim // 2, 3)
            )
        else:
            self.mlp_query = nn.Embedding(query_num, decoder.embed_dim)
            self.plane_pred = nn.Sequential(
                nn.Linear(decoder.embed_dim, decoder.embed_dim // 2),
                nn.GELU(),
                nn.Linear(decoder.embed_dim // 2, 3)
            )
        # assert decoder.embed_dim == encoder.embed_dim
        if decoder.embed_dim == encoder.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder.embed_dim, decoder.embed_dim)
        self.decoder = PointTransformerDecoderEntry(decoder)
        if self.query_ranking:
            self.plane_mlp2 = nn.Sequential(nn.Linear(encoder.embed_dim, encoder.embed_dim),
                                            nn.GELU())
            self.query_ranking = nn.Sequential(
                nn.Linear(decoder.embed_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs, _, _ = xyz.size()
        coor, f, normal, batch = self.grouper(xyz, self.num_centers)
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        x = self.encoder(x + pe, coor)
        # from point proxy to plane proxy
        normal_embed = self.plane_embed(normal)
        x = torch.cat([x, normal_embed], dim=-1)
        plane_encoder = torch.zeros(bs, self.num_planes, x.size(-1)).cuda()
        for i in range(bs):
            unique_batch = torch.unique(batch[i])
            for j, ub in enumerate(unique_batch):
                plane_encoder[i][j] = x[i][(batch[i] == ub).squeeze()].sum(dim=0)
        x = self.plane_mlp(plane_encoder)
        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        mem = self.mem_link(x)
        f_plane = torch.zeros(bs, self.num_planes, normal_embed.size(-1)).cuda()
        for i in range(bs):
            unique_batch = torch.unique(batch[i])
            f_plane[i][:unique_batch.shape[0]] = x[i][:unique_batch.shape[0]]
        if self.query_type == 'dynamic':
            plane = self.plane_pred_coarse(global_feature).reshape(bs, -1, 3)
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, plane.size(1), -1),
                    plane], dim=-1))
            if self.query_ranking:
                f_plane = self.plane_mlp2(f_plane)
                q = torch.cat([q, f_plane], dim=1)
                query_ranking = self.query_ranking(q)
                idx = torch.argsort(query_ranking, dim=1)
                q = torch.gather(q, 1, idx[:, :self.num_queries].expand(-1, -1, q.size(-1)))
            q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
            plane = self.plane_pred(q).reshape(bs, -1, 3)
        elif self.query_type == 'static':
            q = self.mlp_query.weight
            q = q.unsqueeze(0).expand(bs, -1, -1)
            q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
            plane = self.plane_pred(q).reshape(bs, -1, 3)
        return q, plane


@MODELS.register_module()
class PaCoVoxel(nn.Module):
    """
    PACO model modified to use PVD's voxel transformer and proxy decoder with enhanced losses
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.decoder.embed_dim
        self.num_queries = config.num_queries
        self.num_points = getattr(config, 'num_points', None)
        self.decoder_type = config.decoder_type
        self.fold_step = 8
        
        # Calculate factor for point generation
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_queries
            else:
                self.factor = self.fold_step ** 2
        
        # Store factor in config for loss computation
        config.factor = self.factor
        
        # Use enhanced PVD-based pipeline
        self.base_model = PaCoVoxelPipeline(config)
        
        # Enhanced loss function with repulsion and fine-grained chamfer losses
        self.loss_fn = PaCoVoxelLoss(config)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        """
        Forward pass through enhanced PACO-PVD pipeline
        
        Args:
            xyz: (B, N, 3) input point clouds
            
        Returns:
            outputs: dict containing ret, class_prob, proxy_outputs, q
        """
        outputs = self.base_model(xyz)
        return outputs

    def get_loss(self, config, outputs, class_prob, gt, gt_index, plane, plan_index, gt_planes=None):
        """
        Compute enhanced loss using all PACO and PVD loss components
        
        Args:
            config: Configuration with loss weights
            outputs: dict from forward pass
            class_prob: Classification probabilities (from outputs['class_prob'])
            gt: Ground truth point clouds (B, N, 3)
            gt_index: Ground truth plane indices (B, N)
            plane: Ground truth plane parameters (B, P, 4)
            plan_index: Ground truth plane assignment indices
            gt_planes: Optional dict with PVD-style ground truth for proxy decoder losses
            
        Returns:
            loss_dict: Dictionary of computed losses with PACO-compatible structure
        """
        # Extract input points from the first dimension if needed
        if hasattr(self, '_last_input'):
            points = self._last_input
        else:
            # Use gt as fallback for points (this might need adjustment based on your data format)
            points = gt
        
        # Compute enhanced losses
        total_loss, loss_dict = self.loss_fn(
            outputs=outputs,
            gt_planes=gt_planes,
            points=points,
            gt_labels=None,
            gt=gt,
            gt_index=gt_index,
            plane=plane
        )
        
        # Rename losses to match PACO's expected format
        paco_losses = {}
        
        # Map enhanced losses to PACO format
        if 'cls_loss' in loss_dict:
            paco_losses['classification_loss'] = loss_dict['cls_loss']
        if 'param_loss' in loss_dict:
            paco_losses['plane_normal_loss'] = loss_dict['param_loss']
        if 'pvd_chamfer_loss' in loss_dict:
            paco_losses['plane_chamfer_loss'] = loss_dict['pvd_chamfer_loss']
        if 'repulsion_loss' in loss_dict:
            paco_losses['repulsion_loss'] = loss_dict['repulsion_loss']
        if 'chamfer_norm1_loss' in loss_dict:
            paco_losses['chamfer_norm1_loss'] = loss_dict['chamfer_norm1_loss']
        if 'chamfer_norm2_loss' in loss_dict:
            paco_losses['chamfer_norm2_loss'] = loss_dict['chamfer_norm2_loss']
        if 'confidence_loss' in loss_dict:
            paco_losses['confidence_loss'] = loss_dict['confidence_loss']
        
        # Ensure total_loss is included
        paco_losses['total_loss'] = total_loss
        
        return paco_losses

    def compute_loss(self, pred, gt, prefix='', xyz=None):
        """
        Alternative compute_loss method for compatibility
        
        Args:
            pred: outputs from forward pass
            gt: ground truth data
            prefix: prefix for loss names
            xyz: input point clouds
            
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of losses
        """
        # Store input for loss computation
        if xyz is not None:
            self._last_input = xyz
        
        # Prepare gt_planes for PVD losses if available
        gt_planes = None
        if isinstance(gt, dict) and 'planes' in gt:
            gt_planes = gt['planes']
        
        # Extract required ground truth components
        gt_points = gt.get('points', gt) if isinstance(gt, dict) else gt
        gt_index = gt.get('gt_index', torch.zeros_like(gt_points[..., 0]).long()) if isinstance(gt, dict) else torch.zeros_like(gt_points[..., 0]).long()
        plane = gt.get('plane', torch.zeros(gt_points.shape[0], 20, 4).to(gt_points.device)) if isinstance(gt, dict) else torch.zeros(gt_points.shape[0], 20, 4).to(gt_points.device)
        plan_index = gt.get('plan_index', torch.zeros_like(gt_index)) if isinstance(gt, dict) else torch.zeros_like(gt_index)
        
        # Use get_loss method
        config = self.config
        loss_dict = self.get_loss(
            config=config,
            outputs=pred,
            class_prob=pred.get('class_prob', torch.zeros(gt_points.shape[0], self.num_queries, 2).to(gt_points.device)),
            gt=gt_points,
            gt_index=gt_index,
            plane=plane,
            plan_index=plan_index,
            gt_planes=gt_planes
        )
        
        total_loss = loss_dict['total_loss']
        
        # Add prefix to loss names
        if prefix:
            loss_dict = {f'{prefix}_{k}': v for k, v in loss_dict.items()}
            
        return total_loss, loss_dict