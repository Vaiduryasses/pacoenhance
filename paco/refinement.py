"""
Memory-optimized Residual fine-tuning and SVD projection after primitive selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter_max
from torch.utils.checkpoint import checkpoint


class MemoryEfficientEdgeConvBlock(nn.Module):
    """
    Memory-efficient EdgeConv block for residual network
    """
    
    def __init__(self, in_dim, out_dim, k=16, chunk_size=1024):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size
        self.conv = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),  # Use inplace operations
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, pos):
        """
        Args:
            x: (N, C) Point features
            pos: (N, 3) Point positions
        Returns:
            out: (N, out_dim) Updated point features
        """
        N = pos.shape[0]
        device = pos.device
        
        # Process in chunks if too many points
        if N > self.chunk_size:
            return self._forward_chunked(x, pos)
        
        # Compute KNN graph with reduced k if necessary
        effective_k = min(self.k, N - 1)
        if effective_k <= 0:
            # Fallback for very small point clouds
            return torch.zeros(N, self.conv[-1].out_features, device=device)
        
        batch = torch.zeros(N, dtype=torch.long, device=device)
        edge_index = torch_cluster.knn_graph(pos, k=effective_k, batch=batch, loop=False)

        # Get features for source and target nodes
        x_j = x[edge_index[1]]  # Target node features
        x_i = x[edge_index[0]]  # Source node features
        
        # Compute edge features
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        
        # Apply MLPs with gradient checkpointing
        edge_features = checkpoint(self.conv, edge_features, use_reentrant=False)
        
        # Aggregate features using scatter_max
        out, _ = scatter_max(edge_features, edge_index[0], dim=0, dim_size=N)
        
        # Clear intermediate tensors
        del edge_index, x_j, x_i, edge_features
        
        return out
    
    def _forward_chunked(self, x, pos):
        """Process large point clouds in chunks"""
        N = pos.shape[0]
        device = pos.device
        out_dim = self.conv[-1].out_features
        
        # Initialize output
        out = torch.zeros(N, out_dim, device=device)
        
        # Process in overlapping chunks to maintain connectivity
        overlap = self.k
        step = self.chunk_size - overlap
        
        for start in range(0, N, step):
            end = min(start + self.chunk_size, N)
            
            # Extract chunk
            chunk_pos = pos[start:end]
            chunk_x = x[start:end]
            
            # Process chunk
            chunk_out = self._process_chunk(chunk_x, chunk_pos)
            
            # Store results (avoid overlap regions for consistency)
            actual_end = min(start + step, N) if end < N else N
            out[start:actual_end] = chunk_out[:actual_end-start]
            
            # Clear chunk data
            del chunk_pos, chunk_x, chunk_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out
    
    def _process_chunk(self, x, pos):
        """Process a single chunk"""
        N = pos.shape[0]
        effective_k = min(self.k, N - 1)
        
        if effective_k <= 0:
            return torch.zeros(N, self.conv[-1].out_features, device=pos.device)
        
        batch = torch.zeros(N, dtype=torch.long, device=pos.device)
        edge_index = torch_cluster.knn_graph(pos, k=effective_k, batch=batch, loop=False)
        
        x_j = x[edge_index[1]]
        x_i = x[edge_index[0]]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        edge_features = self.conv(edge_features)
        
        out, _ = scatter_max(edge_features, edge_index[0], dim=0, dim_size=N)
        return out


class MemoryEfficientResidualNetwork(nn.Module):
    """
    Memory-efficient residual network for fine-tuning dense points
    """
    
    def __init__(self, k=16, chunk_size=1024):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size

        # EdgeConv blocks with reduced memory footprint
        self.edge_conv1 = MemoryEfficientEdgeConvBlock(3, 64, k=k, chunk_size=chunk_size)
        self.edge_conv2 = MemoryEfficientEdgeConvBlock(64, 64, k=k, chunk_size=chunk_size)
        self.edge_conv3 = MemoryEfficientEdgeConvBlock(64, 64, k=k, chunk_size=chunk_size)
                
        # MLP for final residual prediction
        self.mlp = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
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
        device = points.device
        
        # Process batch efficiently
        if B == 1:
            return self._forward_single(points[0]).unsqueeze(0)
        
        # For larger batches, process with gradient checkpointing
        residuals = []
        for b in range(B):
            pts = points[b]  # (N, 3)
            residual = checkpoint(self._forward_single, pts, use_reentrant=False)
            residuals.append(residual)
            
            # Clear intermediate results
            del pts, residual
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.stack(residuals, dim=0)
    
    def _forward_single(self, pts):
        """Process a single point cloud"""
        # Extract features with mixed precision
        with torch.cuda.amp.autocast():
            f1 = self.edge_conv1(pts, pts)
            f2 = self.edge_conv2(f1, pts)
            f3 = self.edge_conv3(f2, pts)

            # Concatenate features
            features = torch.cat([f1, f2, f3], dim=1)

        # Predict residual (keep in full precision for stability)
        residual = self.mlp(features)
        
        # Clear intermediate features
        del f1, f2, f3, features
        
        return residual


class MemoryEfficientSVDPlaneProjection(nn.Module):
    """
    Memory-efficient SVD-based plane projection
    """
    
    def __init__(self, threshold=0.01, max_points_per_plane=2048):
        super().__init__()
        self.threshold = threshold
        self.max_points_per_plane = max_points_per_plane

    def forward(self, points, planes):
        """
        Args:
            points: (B, N, 3) Input point cloud batch
            planes: (B, M, 4) Plane parameters [a, b, c, d] where ax+by+cz+d=0
        Returns:
            projected_points: (B, N, 3) Points projected to planes
            displacement: (B, N, 3) L2 displacement for loss computation
        """
        B, N, _ = points.shape
        B_p, M, _ = planes.shape
        assert B == B_p, "Batch size mismatch between points and planes"
        
        # Use gradient checkpointing for memory efficiency
        if B == 1:
            projected_points, displacement = self._process_single_batch(
                points[0], planes[0]
            )
            return projected_points.unsqueeze(0), displacement.unsqueeze(0)
        
        # Process multiple batches
        all_projected = []
        all_displacement = []
        
        for b in range(B):
            proj, disp = checkpoint(
                self._process_single_batch, 
                points[b], planes[b], 
                use_reentrant=False
            )
            all_projected.append(proj)
            all_displacement.append(disp)
            
            # Clear intermediate results
            del proj, disp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        projected_points = torch.stack(all_projected, dim=0)
        displacement = torch.stack(all_displacement, dim=0)
        
        return projected_points, displacement
    
    def _process_single_batch(self, pts, batch_planes):
        """Process a single batch item"""
        N, _ = pts.shape
        M, _ = batch_planes.shape
        device = pts.device
        
        projected_points = pts.clone()
        original_points = pts.clone()
        
        # Process each plane
        for i in range(M):
            plane_params = batch_planes[i]
            normal = plane_params[:3]
            distance = plane_params[3]
            
            # Skip invalid planes
            if torch.norm(normal) < 1e-6:
                continue
            
            # Normalize normal vector
            normal = normal / torch.norm(normal)
            
            # Compute point-to-plane distance efficiently
            point_distance = torch.abs(
                torch.sum(pts * normal.unsqueeze(0), dim=1) + distance
            )
            
            # Select points close to the plane
            mask = point_distance < self.threshold
            plane_points = pts[mask]
            
            # Need at least 3 points for SVD
            if plane_points.shape[0] < 3:
                continue
            
            # Subsample if too many points
            if plane_points.shape[0] > self.max_points_per_plane:
                indices = torch.randperm(plane_points.shape[0])[:self.max_points_per_plane]
                plane_points = plane_points[indices]
            
            # Apply SVD projection efficiently
            self._apply_svd_projection(
                projected_points, mask, plane_points, normal, pts
            )
        
        # Compute displacement
        displacement = projected_points - original_points
        
        return projected_points, displacement
    
    def _apply_svd_projection(self, projected_points, mask, plane_points, normal, original_pts):
        """Apply SVD projection to plane points"""
        try:
            # Compute centroid
            centroid = torch.mean(plane_points, dim=0)
            
            # Center points
            centered_points = plane_points - centroid
            
            # Compute covariance matrix efficiently
            cov = torch.mm(centered_points.T, centered_points)
            
            # SVD for plane fitting
            U, S, V = torch.linalg.svd(cov)
            
            # Extract refined normal (eigenvector with smallest eigenvalue)
            refined_normal = V[:, 2]
            
            # Ensure normal points in the same general direction
            if torch.sum(refined_normal * normal) < 0:
                refined_normal = -refined_normal
            
            # Compute refined distance
            refined_distance = -torch.sum(centroid * refined_normal)
            
            # Project inlier points to the refined plane (vectorized)
            mask_indices = torch.where(mask)[0]
            if len(mask_indices) > 0:
                inlier_points = projected_points[mask_indices]
                dot_products = torch.sum(inlier_points * refined_normal.unsqueeze(0), dim=1) + refined_distance
                projections = refined_normal.unsqueeze(0) * dot_products.unsqueeze(1)
                projected_points[mask_indices] = inlier_points - projections
                
        except Exception as e:
            # Skip if SVD fails
            pass


class MemoryEfficientPacoRefinementModule(nn.Module):
    """
    Memory-efficient refinement module for PACO
    """
    
    def __init__(self, residual_knn=16, plane_proj_threshold=0.01, 
                 chunk_size=1024, max_points_per_plane=2048):
        super().__init__()
        self.residual_net = MemoryEfficientResidualNetwork(
            k=residual_knn, chunk_size=chunk_size
        )
        self.plane_projection = MemoryEfficientSVDPlaneProjection(
            threshold=plane_proj_threshold, 
            max_points_per_plane=max_points_per_plane
        )

    def forward(self, points, planes):
        """
        Args:
            points: (B, N, 3) Input point cloud from PACO
            planes: (B, M, 4) Predicted plane parameters from PACO
        Returns:
            dict containing refined results
        """
        # Clear cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute residual with memory efficiency
        with torch.cuda.amp.autocast():
            residual = self.residual_net(points)
        
        # Apply residual
        points_with_residual = points + residual
        
        # Apply SVD-based plane projection
        refined_points, displacement = self.plane_projection(points_with_residual, planes)
        
        # Clear intermediate results
        del points_with_residual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'refined_points': refined_points,
            'residual': residual,
            'displacement': displacement,
            'refined_planes': planes  
        }


# Keep the loss functions unchanged
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