"""
Fixed Adaptive optimized Refinement module with proper AMP and checkpointing support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter_max, scatter_mean
from torch.utils.checkpoint import checkpoint
import math


class FixedAdaptiveEdgeConvBlock(nn.Module):
    """
    修复了混合精度和梯度检查点问题的EdgeConv block
    """
    
    def __init__(self, in_dim, out_dim, k=16, base_chunk_size=4096, memory_efficient=True):
        super().__init__()
        self.k = k
        self.base_chunk_size = base_chunk_size
        self.memory_efficient = memory_efficient
        self.conv = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
        
        # 自适应参数
        self.adaptive_threshold = 2.0
        
    def _get_adaptive_chunk_size(self, N, batch_context_size=1):
        """根据点数和批次上下文动态调整chunk size"""
        memory_factor = 1.0
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            memory_utilization = allocated_memory / total_memory
            
            if memory_utilization > 0.8:
                memory_factor = 0.5
            elif memory_utilization > 0.6:
                memory_factor = 0.7
            elif memory_utilization < 0.3:
                memory_factor = 1.5
        
        batch_factor = max(0.5, 1.0 / math.sqrt(batch_context_size))
        adaptive_size = int(self.base_chunk_size * memory_factor * batch_factor)
        
        min_chunk = 1024
        max_chunk = min(16384, N // 2) if N > 2048 else N
        
        return max(min_chunk, min(adaptive_size, max_chunk))
    
    def forward(self, x, pos, batch_context_size=1):
        """
        Args:
            x: (N, C) Point features
            pos: (N, 3) Point positions
            batch_context_size: 当前处理的batch大小上下文
        Returns:
            out: (N, out_dim) Updated point features
        """
        N = pos.shape[0]
        device = pos.device
        
        # 动态调整chunk size
        chunk_size = self._get_adaptive_chunk_size(N, batch_context_size)
        
        # 如果点数不大，直接处理
        if N <= chunk_size * self.adaptive_threshold:
            return self._forward_direct(x, pos)
        
        # 使用优化的分块处理（不使用梯度检查点避免形状不匹配）
        return self._forward_chunked_adaptive(x, pos, chunk_size)
    
    def _forward_direct(self, x, pos):
        """直接处理，不分块，不使用混合精度避免形状问题"""
        N = pos.shape[0]
        device = pos.device
        
        effective_k = min(self.k, N - 1)
        if effective_k <= 0:
            return torch.zeros(N, self.conv[-1].out_features, device=device)
        
        # 固定batch创建，确保形状一致
        batch = torch.zeros(N, dtype=torch.long, device=device)
        edge_index = torch_cluster.knn_graph(pos, k=effective_k, batch=batch, loop=False)

        # 确保edge_index形状正确
        if edge_index.size(1) == 0:
            return torch.zeros(N, self.conv[-1].out_features, device=device)

        x_j = x[edge_index[1]]
        x_i = x[edge_index[0]]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        
        # 不在这里使用混合精度，避免梯度检查点问题
        edge_features = self.conv(edge_features)
        
        out, _ = scatter_max(edge_features, edge_index[0], dim=0, dim_size=N)
        return out
    
    def _forward_chunked_adaptive(self, x, pos, chunk_size):
        """自适应分块处理，避免梯度检查点"""
        N = pos.shape[0]
        device = pos.device
        out_dim = self.conv[-1].out_features
        
        out = torch.zeros(N, out_dim, device=device)
        
        # 动态调整重叠大小
        overlap = min(self.k, chunk_size // 8)
        step = chunk_size - overlap
        
        # 预计算所有chunk的索引
        chunk_indices = []
        for start in range(0, N, step):
            end = min(start + chunk_size, N)
            actual_end = min(start + step, N) if end < N else N
            chunk_indices.append((start, end, actual_end))
        
        # 批量处理chunks
        for i, (start, end, actual_end) in enumerate(chunk_indices):
            chunk_pos = pos[start:end]
            chunk_x = x[start:end]
            
            chunk_out = self._process_chunk_deterministic(chunk_x, chunk_pos)
            out[start:actual_end] = chunk_out[:actual_end-start]
            
            # 智能内存清理
            if self.memory_efficient and i % max(4, len(chunk_indices) // 8) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return out
    
    def _process_chunk_deterministic(self, x, pos):
        """确定性的chunk处理，避免形状不匹配"""
        N = pos.shape[0]
        effective_k = min(self.k, N - 1)
        
        if effective_k <= 0:
            return torch.zeros(N, self.conv[-1].out_features, device=pos.device)
        
        # 确保批次标识一致
        batch = torch.zeros(N, dtype=torch.long, device=pos.device)
        edge_index = torch_cluster.knn_graph(pos, k=effective_k, batch=batch, loop=False)
        
        # 检查edge_index有效性
        if edge_index.size(1) == 0:
            return torch.zeros(N, self.conv[-1].out_features, device=pos.device)
        
        x_j = x[edge_index[1]]
        x_i = x[edge_index[0]]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        
        # 直接计算，不使用混合精度
        edge_features = self.conv(edge_features)
        
        out, _ = scatter_max(edge_features, edge_index[0], dim=0, dim_size=N)
        return out


class FixedAdaptiveBatchResidualNetwork(nn.Module):
    """
    修复的自适应批处理残差网络，正确处理AMP和梯度检查点
    """
    
    def __init__(self, k=16, base_chunk_size=4096, memory_efficient=True):
        super().__init__()
        self.k = k
        self.base_chunk_size = base_chunk_size
        self.memory_efficient = memory_efficient

        self.edge_conv1 = FixedAdaptiveEdgeConvBlock(3, 64, k=k, base_chunk_size=base_chunk_size, memory_efficient=memory_efficient)
        self.edge_conv2 = FixedAdaptiveEdgeConvBlock(64, 64, k=k, base_chunk_size=base_chunk_size, memory_efficient=memory_efficient)
        self.edge_conv3 = FixedAdaptiveEdgeConvBlock(64, 64, k=k, base_chunk_size=base_chunk_size, memory_efficient=memory_efficient)
                
        self.mlp = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
    
    def forward(self, points):
        """
        Args:
            points: (B, N, 3) Input point cloud batch
        Returns:
            residual: (B, N, 3) Predicted residual batch
        """
        B, N, _ = points.shape
        
        # 根据batch size选择策略，避免梯度检查点问题
        return self._forward_safe_strategy(points)
    
    def _forward_safe_strategy(self, points):
        """安全的处理策略，避免AMP和梯度检查点冲突"""
        B, N, _ = points.shape
        
        if B == 1:
            # 单样本直接处理
            return self._forward_single_safe(points[0], B).unsqueeze(0)
        elif B <= 8:
            # 小batch使用混合精度但不用梯度检查点
            return self._forward_small_batch_amp(points)
        elif B <= 20:
            # 中等batch分组处理，使用梯度检查点但不用混合精度
            return self._forward_medium_batch_checkpoint(points)
        else:
            # 大batch使用最保守的策略
            return self._forward_large_batch_conservative(points)
    
    def _forward_small_batch_amp(self, points):
        """小batch使用混合精度但不用梯度检查点"""
        B = points.shape[0]
        residuals = []
        
        # 对整个batch使用混合精度
        with torch.cuda.amp.autocast():
            for b in range(B):
                residual = self._forward_single_safe(points[b], B)
                residuals.append(residual)
        
        return torch.stack(residuals, dim=0)
    
    def _forward_medium_batch_checkpoint(self, points):
        """中等batch使用梯度检查点但不用混合精度"""
        B, N, _ = points.shape
        
        # 动态确定组大小
        group_size = max(2, min(8, 32 // max(1, N // 2048)))
        
        all_residuals = []
        
        for start in range(0, B, group_size):
            end = min(start + group_size, B)
            group_points = points[start:end]
            
            group_residuals = []
            for b in range(group_points.shape[0]):
                # 使用梯度检查点但不用混合精度
                residual = checkpoint(
                    self._forward_single_no_amp, 
                    group_points[b], 
                    end - start, 
                    use_reentrant=False
                )
                group_residuals.append(residual)
            
            all_residuals.extend(group_residuals)
            
            # 智能内存管理
            if self.memory_efficient and start % (group_size * 4) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return torch.stack(all_residuals, dim=0)
    
    def _forward_large_batch_conservative(self, points):
        """大batch使用最保守的策略"""
        B, N, _ = points.shape
        
        # 分组处理，既不用混合精度也不用梯度检查点
        group_size = max(1, 16 // max(1, N // 1024))
        
        residuals = []
        for start in range(0, B, group_size):
            end = min(start + group_size, B)
            
            for b in range(start, end):
                residual = self._forward_single_no_amp(points[b], B)
                residuals.append(residual)
            
            # 每组处理完后清理内存
            if self.memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.stack(residuals, dim=0)
    
    def _forward_single_safe(self, pts, batch_context_size=1):
        """安全的单样本处理，可以配合混合精度使用"""
        # 这里已经在混合精度上下文中，直接计算
        f1 = self.edge_conv1(pts, pts, batch_context_size)
        f2 = self.edge_conv2(f1, pts, batch_context_size)
        f3 = self.edge_conv3(f2, pts, batch_context_size)
        features = torch.cat([f1, f2, f3], dim=1)

        # MLP部分保持全精度
        residual = self.mlp(features.float())
        return residual
    
    def _forward_single_no_amp(self, pts, batch_context_size=1):
        """无混合精度的单样本处理，适合梯度检查点"""
        f1 = self.edge_conv1(pts, pts, batch_context_size)
        f2 = self.edge_conv2(f1, pts, batch_context_size)
        f3 = self.edge_conv3(f2, pts, batch_context_size)
        features = torch.cat([f1, f2, f3], dim=1)

        residual = self.mlp(features)
        return residual


class FixedAdaptiveSVDPlaneProjection(nn.Module):
    """
    修复的自适应SVD平面投影，避免梯度检查点问题
    """
    
    def __init__(self, threshold=0.01, base_max_points_per_plane=4096, 
                 base_svd_batch_size=16, memory_efficient=True):
        super().__init__()
        self.threshold = threshold
        self.base_max_points_per_plane = base_max_points_per_plane
        self.base_svd_batch_size = base_svd_batch_size
        self.memory_efficient = memory_efficient

    def _get_adaptive_params(self, B, N, M):
        """根据输入规模自适应调整参数"""
        total_operations = B * N * M
        
        if total_operations > 500000:
            max_points_factor = 0.5
        elif total_operations > 200000:
            max_points_factor = 0.7
        else:
            max_points_factor = 1.0
        
        max_points_per_plane = int(self.base_max_points_per_plane * max_points_factor)
        svd_batch_size = min(self.base_svd_batch_size, max(4, M // 2))
        
        if B <= 8:
            group_size = B
        elif B <= 20:
            group_size = 8
        elif B <= 50:
            group_size = 10
        else:
            group_size = max(8, B // 8)
        
        return max_points_per_plane, svd_batch_size, group_size

    def forward(self, points, planes):
        """
        Args:
            points: (B, N, 3) Input point cloud batch
            planes: (B, M, 4) Plane parameters
        Returns:
            projected_points: (B, N, 3) Points projected to planes
            displacement: (B, N, 3) L2 displacement for loss computation
        """
        B, N, _ = points.shape
        M = planes.shape[1]
        
        # 获取自适应参数
        max_points_per_plane, svd_batch_size, group_size = self._get_adaptive_params(B, N, M)
        
        if B == 1:
            projected_points, displacement = self._process_single_deterministic(
                points[0], planes[0], max_points_per_plane, svd_batch_size
            )
            return projected_points.unsqueeze(0), displacement.unsqueeze(0)
        else:
            return self._process_adaptive_batch_safe(
                points, planes, max_points_per_plane, svd_batch_size, group_size
            )
    
    def _process_adaptive_batch_safe(self, points, planes, max_points_per_plane, 
                                   svd_batch_size, group_size):
        """安全的自适应批处理，避免梯度检查点问题"""
        B = points.shape[0]
        
        all_projected = []
        all_displacement = []
        
        for start in range(0, B, group_size):
            end = min(start + group_size, B)
            
            group_proj = []
            group_disp = []
            
            for b in range(start, end):
                # 避免使用梯度检查点，直接处理
                proj, disp = self._process_single_deterministic(
                    points[b], planes[b], max_points_per_plane, svd_batch_size
                )
                
                group_proj.append(proj)
                group_disp.append(disp)
            
            all_projected.extend(group_proj)
            all_displacement.extend(group_disp)
            
            # 智能内存清理
            if self.memory_efficient and start % (group_size * 2) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return torch.stack(all_projected, dim=0), torch.stack(all_displacement, dim=0)
    
    def _process_single_deterministic(self, pts, batch_planes, max_points_per_plane, svd_batch_size):
        """确定性的单样本处理，避免形状不匹配"""
        N, _ = pts.shape
        M, _ = batch_planes.shape
        device = pts.device
        
        projected_points = pts.clone()
        original_points = pts.clone()
        
        # 预计算和过滤有效平面
        normals = batch_planes[:, :3]
        distances = batch_planes[:, 3]
        
        valid_mask = torch.norm(normals, dim=1) > 1e-6
        if not valid_mask.any():
            return projected_points, projected_points - original_points
        
        valid_normals = F.normalize(normals[valid_mask], dim=1)
        valid_distances = distances[valid_mask]
        
        # 批量计算距离
        point_distances = torch.abs(
            torch.mm(pts, valid_normals.T) + valid_distances.unsqueeze(0)
        )
        
        # 确定性批量处理平面
        self._batch_process_planes_deterministic(
            projected_points, pts, valid_normals, valid_distances, 
            point_distances, max_points_per_plane, svd_batch_size
        )
        
        displacement = projected_points - original_points
        return projected_points, displacement
    
    def _batch_process_planes_deterministic(self, projected_points, original_pts, normals, 
                                          distances, point_distances, max_points_per_plane, svd_batch_size):
        """确定性批量处理平面"""
        num_planes = normals.shape[0]
        
        for start_idx in range(0, num_planes, svd_batch_size):
            end_idx = min(start_idx + svd_batch_size, num_planes)
            
            batch_normals = normals[start_idx:end_idx]
            batch_distances = distances[start_idx:end_idx]
            batch_point_dists = point_distances[:, start_idx:end_idx]
            
            for i, (normal, distance) in enumerate(zip(batch_normals, batch_distances)):
                plane_point_dists = batch_point_dists[:, i]
                
                mask = plane_point_dists < self.threshold
                if mask.sum() < 3:
                    continue
                
                plane_points = original_pts[mask]
                
                # 确定性采样
                if plane_points.shape[0] > max_points_per_plane:
                    # 使用确定性采样而不是随机采样
                    step = plane_points.shape[0] // max_points_per_plane
                    indices = torch.arange(0, plane_points.shape[0], step, device=plane_points.device)[:max_points_per_plane]
                    plane_points = plane_points[indices]
                
                # SVD拟合和投影
                refined_normal, refined_distance = self._robust_svd_fitting(plane_points, normal)
                self._vectorized_projection(projected_points, mask, refined_normal, refined_distance)
    
    def _robust_svd_fitting(self, plane_points, initial_normal):
        """鲁棒的SVD平面拟合"""
        try:
            centroid = plane_points.mean(dim=0)
            centered = plane_points - centroid
            
            n_points = plane_points.shape[0]
            cov = torch.mm(centered.T, centered) / max(1, n_points - 1)
            
            # 添加正则化防止奇异
            reg_term = torch.eye(3, device=cov.device) * 1e-6
            cov = cov + reg_term
            
            U, S, V = torch.linalg.svd(cov)
            refined_normal = V[:, 2]
            
            # 保持方向一致
            if torch.dot(refined_normal, initial_normal) < 0:
                refined_normal = -refined_normal
            
            refined_distance = -torch.dot(centroid, refined_normal)
            return refined_normal, refined_distance
            
        except:
            # SVD失败时使用原始法向量
            centroid = plane_points.mean(dim=0)
            return initial_normal, -torch.dot(centroid, initial_normal)
    
    def _vectorized_projection(self, projected_points, mask, normal, distance):
        """向量化投影操作"""
        mask_indices = torch.where(mask)[0]
        if len(mask_indices) == 0:
            return
        
        points_to_project = projected_points[mask_indices]
        dot_products = torch.sum(points_to_project * normal.unsqueeze(0), dim=1) + distance
        projections = normal.unsqueeze(0) * dot_products.unsqueeze(1)
        projected_points[mask_indices] = points_to_project - projections


class FixedAdaptivePacoRefinementModule(nn.Module):
    """
    修复的自适应PACO精炼模块，解决AMP和梯度检查点冲突
    """
    
    def __init__(self, residual_knn=8, plane_proj_threshold=0.01, 
                 base_chunk_size=1024, base_max_points_per_plane=1024, 
                 base_svd_batch_size=8, memory_efficient=True):
        super().__init__()
        
        self.residual_net = FixedAdaptiveBatchResidualNetwork(
            k=residual_knn, 
            base_chunk_size=base_chunk_size,
            memory_efficient=memory_efficient
        )
        
        self.plane_projection = FixedAdaptiveSVDPlaneProjection(
            threshold=plane_proj_threshold, 
            base_max_points_per_plane=base_max_points_per_plane,
            base_svd_batch_size=base_svd_batch_size,
            memory_efficient=memory_efficient
        )
        
        self.memory_efficient = memory_efficient

    def forward(self, points, planes):
        """
        Args:
            points: (B, N, 3) Input point cloud from PACO
            planes: (B, M, 4) Predicted plane parameters from PACO
        Returns:
            dict containing refined results
        """
        B = points.shape[0]
        
        # 智能内存管理
        if self.memory_efficient and B > 8 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 不在这里使用混合精度，让子模块自己决定
        residual = self.residual_net(points)
        
        # 应用残差
        points_with_residual = points + residual
        
        # 应用SVD投影
        refined_points, displacement = self.plane_projection(points_with_residual, planes)
        
        return {
            'refined_points': refined_points,
            'residual': residual,
            'displacement': displacement,
            'refined_planes': planes  
        }


# 保持损失函数不变
class ResidualLoss(nn.Module):
    """MSE loss for residual refinement"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_residual, target_residual):
        return F.mse_loss(pred_residual, target_residual)


class SVDProjectionLoss(nn.Module):
    """L2 loss for SVD projection displacement"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, displacement):
        return torch.mean(torch.norm(displacement, p=2, dim=-1))