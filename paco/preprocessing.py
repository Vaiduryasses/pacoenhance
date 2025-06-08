"""
Preprocessing functions from PVD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_sparse
import numpy as np


def normalize_pointcloud(points):
    """
    Center and scale pointcloud to unit cube
    Args:
        points: Tensor of shape (N, 3)
    Returns:
        normalized points, centroid, scale
    """
    if isinstance(points, np.ndarray):
        centroid = points.mean(0)
        points = points - centroid
        scale = points.ptp()  # Max range across all dimensions
        points = points / scale
        return points, centroid, scale
    else: 
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        scale = torch.max(points) - torch.min(points)  # ptp
        points = points / scale
        return points, centroid, scale


def voxelize_pointcloud(points, resolution=64, features=None):
    """
    Voxelize a pointcloud to sparse tensor format
    Args:
        points: Tensor of shape (N, 3) in range [-0.5, 0.5]
        resolution: Voxel grid resolution
        features: Optional point features (N, F)
    Returns:
        voxel_features: Features per occupied voxel (V, F+8)
        voxel_coords: Coordinates of occupied voxels (V, 3)
        sparse_tensor: SparseTensor representation
    """
    scaled_points = (points + 0.5) * (resolution - 1)
    voxel_indices = torch.floor(scaled_points).long()
    voxel_indices = torch.clamp(voxel_indices, 0, resolution - 1)

    # Create voxel hash
    voxel_hash = voxel_indices[:, 0] * resolution**2 + voxel_indices[:, 1] * resolution + voxel_indices[:, 2]
    
    # Find unique voxels and counts
    unique_voxels, inverse_indices, counts = torch.unique(voxel_hash, return_inverse=True, return_counts=True)

    # Prepare voxel coordinates
    num_voxels = len(unique_voxels)
    voxel_coords = torch.zeros((num_voxels, 3), dtype=torch.long, device=points.device)
    
    # Compute voxel coordinates and centroids
    for i, h in enumerate(unique_voxels):
        mask = (voxel_hash == h)
        coords = voxel_indices[mask][0]
        voxel_coords[i] = coords

    # Compute centroids and counts as initial features
    base_features = []

    # Compute centroids by scatter mean
    centroids = scatter_mean(points, inverse_indices, dim=0)
    base_features.append(centroids)

    # Add counts as feature
    log_counts = torch.log(counts.float() + 1) / torch.log(torch.tensor(100.0, device=points.device))
    base_features.append(log_counts.unsqueeze(1))

    # Compute PCA normals as additional feature
    if points.shape[0] > 10:  # Only if we have enough points
        try:
            # For each voxel, compute covariance of points
            voxel_points = []
            for i, h in enumerate(unique_voxels):
                mask = (voxel_hash == h)
                if mask.sum() >= 3:  # Need at least 3 points for PCA
                    voxel_points.append(points[mask] - centroids[i])
                else:
                    voxel_points.append(torch.zeros((3, 3), device=points.device))
            
            # Compute covariance matrices
            covs = []
            for pts in voxel_points:
                if pts.shape[0] >= 3:
                    cov = pts.t() @ pts / (pts.shape[0] - 1)
                    covs.append(cov)
                else:
                    covs.append(torch.eye(3, device=points.device))
            
            # Compute eigenvalues and eigenvectors
            covs = torch.stack(covs)  # (V, 3, 3)
            evals, evecs = torch.linalg.eigh(covs)
            
            # Get normal (eigenvector with smallest eigenvalue)
            normals = evecs[:, :, 0]  # (V, 3)
            variances = evals[:, 0].unsqueeze(1)  # (V, 1)
            
            base_features.append(normals)
            base_features.append(variances)
        except:
            # Fallback if PCA fails
            normals = torch.zeros((num_voxels, 3), device=points.device)
            variances = torch.ones((num_voxels, 1), device=points.device)
            base_features.append(normals)
            base_features.append(variances)

    # Concatenate
    voxel_features = torch.cat(base_features, dim=1)

    # Add point features if provided
    if features is not None:
        point_features = scatter_mean(features, inverse_indices, dim=0)
        voxel_features = torch.cat([voxel_features, point_features], dim=1)
    
    # Create sparse tensor
    indices = voxel_coords.t().contiguous()  # (3, V) for sparse API
    shape = (resolution, resolution, resolution)
    sparse_tensor = torch_sparse.SparseTensor(indices=indices, values=voxel_features, size=shape)
    
    return voxel_features, voxel_coords, sparse_tensor


class FeatureExpansionMLP(nn.Module):
    """
    MLP network to expand 8D features to 32D
    
    """
    
    def __init__(self):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
    
    def forward(self, x):
        """Forward pass to expand features
        
        Args:
            x: Input tensor of shape (N, 8) - 8D voxel features
            
        Returns:
            Expanded features of shape (N, 32)
        """
        return self.mlp(x)


def process_point_cloud(points, features=None):
    """
    Complete point cloud processing pipeline:
    1. Voxelize to get 8D features
    2. Expand features to 32D using MLP
    3. Add 3D original centroids
    4. Output 35D features
    
    Args:
        points: (N, 3) input point cloud
        features: Optional (N, F) point features
        
    Returns:
        voxel_coords: (V, 3) voxel coordinates
        final_features: (V, 35) processed features
    """
    # Normalize point cloud
    normalized_points, center, scale = normalize_pointcloud(points)
    
    # Step 1: Voxelize point cloud to get 8D features
    voxel_features, voxel_coords, _ = voxelize_pointcloud(
        normalized_points, resolution=64, features=features)
    
    # Step 2: Expand to 32D using MLP
    feature_expansion = FeatureExpansionMLP()
    expanded_features = feature_expansion(voxel_features)
    
    # Step 3: Calculate centroids in original coordinate system (3D)
    # Convert from normalized voxel space back to original space
    original_centroids = voxel_coords.float() / (64 - 1)  # [0, 1]
    original_centroids = (original_centroids * 2 - 1) * scale + center  # Back to original space
    
    # Step 4: Concatenate expanded features with original centroids for final 35D
    final_features = torch.cat([expanded_features, original_centroids], dim=1)
    
    return voxel_coords, final_features