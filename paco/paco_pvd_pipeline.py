"""
Modified PACO Pipeline with PVD Voxel Transformer and Proxy Decoder
Enhanced with Complete Loss Functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_sparse
from scipy.optimize import linear_sum_assignment
from ..config import cfg
from .voxel_transformer import SparseVoxelTransformer  # From PVD
from .proxy_decoder import ProxyDecoder  # From PVD
from .preprocessing import voxelize_pointcloud, FeatureExpansionMLP, process_point_cloud  # From PVD


class PaCoVoxelPipeline(nn.Module):
    """
    PACO pipeline modified to use PVD's voxel transformer and proxy decoder
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Store config
        self.config = config
        self.num_queries = config.num_queries
        self.voxel_resolution = getattr(config, 'voxel_resolution', 64)
        
        # PACO-specific parameters for point reconstruction
        self.decoder_type = getattr(config, 'decoder_type', 'fold')
        self.fold_step = 8
        self.factor = self.fold_step ** 2  # Points per query
        self.repulsion = getattr(config, 'repulsion', None)
        
        # PVD Voxel Preprocessing
        self.feature_expansion = FeatureExpansionMLP()
        
        # PVD Voxel Transformer (replaces PCTransformer encoder)
        self.voxel_transformer = SparseVoxelTransformer(
            in_dim=35,  # 32(expanded features) + 3(centroids)
            dim=config.encoder.embed_dim,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            window_size=getattr(config, 'window_size', 3),
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # PVD Proxy Decoder (replaces PointTransformerDecoder)
        self.proxy_decoder = ProxyDecoder(
            dim=config.encoder.embed_dim,
            num_queries=self.num_queries,
            num_layers=getattr(config.decoder, 'depth', 4),
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # Additional projection layers if needed
        if hasattr(config, 'global_feature_dim'):
            self.global_proj = nn.Sequential(
                nn.Linear(config.encoder.embed_dim, config.global_feature_dim),
                nn.ReLU(),
                nn.Linear(config.global_feature_dim, config.encoder.embed_dim)
            )
        else:
            self.global_proj = nn.Identity()
        
        # PACO-style reconstruction heads
        self.increase_dim = nn.Sequential(
            nn.Conv1d(config.encoder.embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        
        self.reduce_map = nn.Linear(config.encoder.embed_dim + 1027, config.encoder.embed_dim)
        hidden_dim = 256
        
        # Point reconstruction decoder
        if self.decoder_type == 'fold':
            self.decode_head = Fold(config.encoder.embed_dim, step=self.fold_step, hidden_dim=256, freedom=2)
        else:
            self.decode_head = SimpleRebuildFCLayer(
                config.encoder.embed_dim * 2,
                step=self.fold_step ** 2,
                freedom=2
            )
        
        self.rebuild_map = nn.Sequential(
            nn.Conv1d(config.encoder.embed_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, 2)
            
        # Plane prediction head (converts proxy outputs to final plane parameters)
        self.plane_pred = nn.Sequential(
            nn.Linear(config.encoder.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # spherical coordinates (theta, phi, r)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def normalize_pointcloud(self, points):
        """Normalize point cloud to [-0.5, 0.5] range"""
        centroid = torch.mean(points, dim=1, keepdim=True)  # (B, 1, 3)
        points_centered = points - centroid
        scale = torch.max(torch.abs(points_centered).reshape(points.shape[0], -1), dim=1)[0]  # (B,)
        scale = scale.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        points_normalized = points_centered / (scale + 1e-8)
        return points_normalized, centroid, scale
    
    def voxelize_batch(self, points_batch):
        """
        Voxelize a batch of point clouds
        Args:
            points_batch: (B, N, 3) batch of point clouds
        Returns:
            batch_voxel_features: list of (V_i, 35) voxel features for each batch
            batch_voxel_coords: list of (V_i, 3) voxel coordinates for each batch
        """
        batch_size = points_batch.shape[0]
        batch_voxel_features = []
        batch_voxel_coords = []
        
        for b in range(batch_size):
            points = points_batch[b]  # (N, 3)
            
            # Process single point cloud
            voxel_coords, voxel_features = process_point_cloud(
                points, features=None
            )
            
            batch_voxel_features.append(voxel_features)
            batch_voxel_coords.append(voxel_coords)
        
        return batch_voxel_features, batch_voxel_coords
    
    def pad_voxel_features(self, batch_voxel_features, batch_voxel_coords, max_voxels=None):
        """
        Pad voxel features to same length for batch processing
        """
        if max_voxels is None:
            max_voxels = max(features.shape[0] for features in batch_voxel_features)
        
        batch_size = len(batch_voxel_features)
        feature_dim = batch_voxel_features[0].shape[1]
        
        # Create padded tensors
        padded_features = torch.zeros(batch_size, max_voxels, feature_dim, 
                                    device=batch_voxel_features[0].device)
        padded_coords = torch.zeros(batch_size, max_voxels, 3,
                                  device=batch_voxel_coords[0].device)
        masks = torch.zeros(batch_size, max_voxels, dtype=torch.bool,
                           device=batch_voxel_features[0].device)
        
        for b in range(batch_size):
            num_voxels = batch_voxel_features[b].shape[0]
            padded_features[b, :num_voxels] = batch_voxel_features[b]
            padded_coords[b, :num_voxels] = batch_voxel_coords[b].float()
            masks[b, :num_voxels] = True
            
        return padded_features, padded_coords, masks
    
    def forward(self, xyz):
        """
        Forward pass through modified PACO pipeline
        
        Args:
            xyz: (B, N, 3) input point clouds
            
        Returns:
            dict with:
                - ret: tuple of (predicted_planes, reconstructed_points) like PACO
                - class_prob: classification probabilities like PACO
                - proxy_outputs: raw outputs from proxy decoder
        """
        batch_size = xyz.shape[0]
        
        # Step 1: Normalize point clouds
        xyz_normalized, centroid, scale = self.normalize_pointcloud(xyz)
        
        # Step 2: Voxelize point clouds using PVD preprocessing
        batch_voxel_features, batch_voxel_coords = self.voxelize_batch(xyz_normalized)
        
        # Step 3: Pad features for batch processing
        voxel_features, voxel_coords, voxel_masks = self.pad_voxel_features(
            batch_voxel_features, batch_voxel_coords
        )
        
        # Step 4: Pass through voxel transformer
        # Reshape for transformer: (B*V, 35) -> apply transformer -> (B*V, dim)
        B, V, F = voxel_features.shape
        voxel_features_flat = voxel_features.reshape(-1, F)  # (B*V, 35)
        voxel_coords_flat = voxel_coords.reshape(-1, 3)      # (B*V, 3)
        
        # Create batch indices for transformer
        batch_indices = torch.arange(B, device=xyz.device).unsqueeze(1).expand(-1, V).reshape(-1)
        
        # Apply voxel transformer
        global_features = self.voxel_transformer(voxel_features_flat, voxel_coords_flat)
        
        # Reshape back: (B*V, dim) -> (B, V, dim)
        global_features = global_features.reshape(B, V, -1)
        
        # Step 5: Global pooling to get context for proxy decoder
        # Mask out padded voxels
        global_features = global_features * voxel_masks.unsqueeze(-1).float()
        
        # Global max pooling
        context_features = torch.max(global_features, dim=1)[0]  # (B, dim)
        
        # Optional projection
        context_features = self.global_proj(context_features)
        
        # Step 6: Pass through proxy decoder
        proxy_outputs = self.proxy_decoder(context_features)
        
        # Step 7: Extract query features and plane parameters
        query_features = proxy_outputs['query_features']  # (B, num_queries, dim)
        plane_spherical = self.plane_pred(query_features)  # (B, num_queries, 3) - spherical coordinates
        
        # Step 8: PACO-style point reconstruction
        B, M, C = query_features.shape
        
        # Global feature processing like PACO
        global_feature = self.increase_dim(query_features.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        
        # Rebuild feature combination like PACO
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            query_features,
            plane_spherical
        ], dim=-1)  # B M (1024 + C + 3)
        
        # Decode points like PACO
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            angle_point = self.decode_head(rebuild_feature).reshape(B, M, 2, -1)  # B M 2 S
            theta_point = angle_point[:, :, 0, :].unsqueeze(2)
            phi_point = angle_point[:, :, 1, :].unsqueeze(2)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            angle_point = self.decode_head(rebuild_feature)  # B M S 2
            theta_point = angle_point[:, :, :, 0]
            phi_point = angle_point[:, :, :, 1]
        
        rebuild_feature = self.rebuild_map(rebuild_feature.reshape(B * M, -1).unsqueeze(-1))
        
        # Generate the plane points like PACO
        theta = plane_spherical[:, :, 0].unsqueeze(-1).expand(-1, M, theta_point.size(2))
        phi = plane_spherical[:, :, 1].unsqueeze(-1).expand(-1, M, theta_point.size(2))
        r = plane_spherical[:, :, 2].unsqueeze(-1).expand(-1, M, theta_point.size(2))
        
        N = torch.cos(phi_point - phi) * torch.sin(theta_point) * torch.sin(theta) + torch.cos(theta_point) * torch.cos(theta)
        N = torch.clamp(N, min=1e-6)
        r2 = r / N
        
        # Point cloud generation
        x_coord = (r2 * torch.sin(theta_point) * torch.cos(phi_point)).unsqueeze(-1)
        y_coord = (r2 * torch.sin(theta_point) * torch.sin(phi_point)).unsqueeze(-1)
        z_coord = (r2 * torch.cos(theta_point)).unsqueeze(-1)
        rebuild_points = torch.cat([x_coord, y_coord, z_coord], dim=-1)
        rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()
        rebuild_points = torch.clamp(rebuild_points, min=-1, max=1)
        
        # Convert spherical to cartesian plane parameters like PACO
        a = torch.sin(plane_spherical[:, :, 0]) * torch.cos(plane_spherical[:, :, 1])
        b = torch.sin(plane_spherical[:, :, 0]) * torch.sin(plane_spherical[:, :, 1])
        c = torch.cos(plane_spherical[:, :, 0])
        d = -plane_spherical[:, :, 2]
        plane_cartesian = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), d.unsqueeze(-1)], dim=-1)
        
        # Classification probabilities like PACO
        class_prob = self.classifier(rebuild_feature.squeeze(-1))  # B*M 2
        class_prob = class_prob.reshape(B, M, -1)
        
        # Return in PACO format
        ret = (plane_cartesian, rebuild_points)
        
        return {
            'ret': ret,
            'class_prob': class_prob,
            'proxy_outputs': proxy_outputs,
            'q': query_features  # For compatibility
        }


class PaCoVoxelLoss(nn.Module):
    """
    Enhanced loss function combining PVD proxy decoder losses with PACO repulsion and fine-grained chamfer losses
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_queries = config.num_queries
        self.factor = getattr(config, 'factor', 64)  # Points per query
        self.repulsion = getattr(config, 'repulsion', None)
        
        # PVD proxy decoder weights
        self.alpha_param = getattr(config, 'alpha_param', 0.5)
        self.beta_chamfer = getattr(config, 'beta_chamfer', 20.0)
        
        # PACO-specific weights
        self.w_classification = getattr(config, 'w_classification', 1.0)
        self.w_confidence = getattr(config, 'w_confidence', 0.5)
        self.w_repulsion = getattr(config, 'w_repulsion', 1.0)
        self.w_chamfer_norm1 = getattr(config, 'w_chamfer_norm1', 1.0)
        self.w_chamfer_norm2 = getattr(config, 'w_chamfer_norm2', 1.0)
        
    def compute_repulsion_loss(self, reconstructed_points, batch_idx):
        """
        Compute repulsion loss to prevent points from clustering too closely
        
        Args:
            reconstructed_points: (B, N*factor, 3) reconstructed points
            batch_idx: current batch index
            
        Returns:
            repulsion_penalty: scalar tensor
        """
        if self.repulsion is None:
            return torch.tensor(0.0, device=reconstructed_points.device)
        
        # Reshape points for repulsion computation
        reshaped_points = reconstructed_points[batch_idx].view(-1, self.factor, 3)
        
        # Import required functions
        try:
            from .transformer_utils import knn_point, index_points
        except ImportError:
            # Fallback simple implementation
            return torch.tensor(0.0, device=reconstructed_points.device)
        
        # Find k-nearest neighbors (excluding self)
        neighbor_indices = knn_point(
            self.repulsion.num_neighbors, reshaped_points, reshaped_points
        )[:, :, 1:].long()
        
        # Get neighbor points
        grouped_points = index_points(reshaped_points, neighbor_indices).transpose(2, 3).contiguous() - reshaped_points.unsqueeze(-1)
        
        # Compute distance matrix
        distance_matrix = torch.sum(grouped_points ** 2, dim=2).clamp(min=self.repulsion.epsilon)
        
        # Compute weights based on distance
        weight_matrix = torch.exp(-distance_matrix / self.repulsion.kernel_bandwidth ** 2)
        
        # Repulsion penalty: penalize points that are closer than radius
        repulsion_penalty = torch.mean(
            (self.repulsion.radius - distance_matrix.sqrt()) * weight_matrix,
            dim=(1, 2)
        ).clamp(min=0)
        
        return repulsion_penalty.mean()
    
    def compute_fine_chamfer_loss(self, matched_reconstructed_points, gt_points, norm=2):
        """
        Compute fine-grained chamfer loss with specified norm
        
        Args:
            matched_reconstructed_points: (1, N, 3) matched reconstructed points
            gt_points: (N, 3) ground truth points
            norm: 1 for L1 norm, 2 for L2 norm
            
        Returns:
            chamfer_loss: scalar tensor
        """
        try:
            from pytorch3d.loss import chamfer_distance
            
            chamfer_loss = chamfer_distance(
                matched_reconstructed_points, 
                gt_points.unsqueeze(0), 
                norm=norm
            )
            return chamfer_loss[0]  # Return the distance value
        except ImportError:
            # Fallback simple chamfer distance implementation
            pred_points = matched_reconstructed_points.squeeze(0)  # (N, 3)
            
            # Compute pairwise distances
            pred_expanded = pred_points.unsqueeze(1)  # (N, 1, 3)
            gt_expanded = gt_points.unsqueeze(0)      # (1, M, 3)
            
            if norm == 1:
                dists = torch.sum(torch.abs(pred_expanded - gt_expanded), dim=2)  # L1 norm
            else:
                dists = torch.sum((pred_expanded - gt_expanded) ** 2, dim=2)      # L2 norm
            
            # Chamfer distance: min distance from each pred to any gt + min distance from each gt to any pred
            chamfer_pred_to_gt = torch.min(dists, dim=1)[0].mean()
            chamfer_gt_to_pred = torch.min(dists, dim=0)[0].mean()
            
            return (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2
        
    def forward(self, outputs, gt_planes, points, gt_labels=None, gt=None, gt_index=None, plane=None):
        """
        Compute enhanced loss combining PVD and PACO approaches
        
        Args:
            outputs: dict from forward pass containing proxy_outputs, ret, class_prob
            gt_planes: dict with ground truth plane parameters for PVD losses
            points: (B, N, 3) input point clouds
            gt_labels: optional ground truth labels
            gt: (B, N, 3) ground truth point clouds for PACO losses
            gt_index: (B, N) ground truth plane indices for PACO losses
            plane: (B, P, 4) ground truth plane parameters for PACO losses
            
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual losses
        """
        proxy_outputs = outputs['proxy_outputs']
        ret = outputs['ret']
        class_prob = outputs['class_prob']
        
        predicted_planes, reconstructed_points = ret
        batch_size = reconstructed_points.shape[0]
        device = reconstructed_points.device
        
        # Initialize losses
        losses = {}
        
        # === PVD-style Proxy Decoder Losses ===
        if gt_planes is not None:
            # Extract predictions from proxy decoder
            pred_logits = proxy_outputs.get('logits', None)
            pred_normals = proxy_outputs.get('normals', None)
            pred_distances = proxy_outputs.get('distances', None)
            
            if pred_logits is not None and pred_normals is not None and pred_distances is not None:
                # Extract ground truth
                gt_normals = gt_planes['normals']    # (B, P', 3)
                gt_distances = gt_planes['distances'] # (B, P')
                gt_masks = gt_planes['masks']        # (B, P', N) - point masks for each plane
                
                # Handle batch dimensions
                if pred_logits.dim() == 1:
                    pred_logits = pred_logits.unsqueeze(0)
                    pred_normals = pred_normals.unsqueeze(0)
                    pred_distances = pred_distances.unsqueeze(0)

                if gt_normals.dim() == 2:
                    gt_normals = gt_normals.unsqueeze(0)
                    gt_distances = gt_distances.unsqueeze(0)
                    gt_masks = gt_masks.unsqueeze(0)
                
                num_pred = pred_logits.shape[1]
                num_gt = gt_normals.shape[1]

                # Initialize PVD losses
                cls_loss = 0
                param_loss = 0
                pvd_chamfer_loss = 0

                # Process each batch using Hungarian matching (from PVD ProxyDecoderLoss)
                for b in range(batch_size):
                    # Compute cost matrix for matching
                    cost_matrix = torch.zeros((num_pred, num_gt), device=device)

                    # Compute cost based on normal and distance similarity
                    for i in range(num_pred):
                        for j in range(num_gt):
                            # Normal similarity (1 - |cos(angle)|)
                            normal_sim = 1 - torch.abs(torch.sum(pred_normals[b, i] * gt_normals[b, j]))

                            # Distance difference
                            dist_diff = torch.abs(pred_distances[b, i] - gt_distances[b, j])

                            # Chamfer distance for points
                            if torch.any(gt_masks[b, j]):
                                # Get points belonging to this plane
                                gt_plane_points = points[b][gt_masks[b, j]]
                                
                                # Compute point-to-plane distance for predicted plane
                                point_to_plane = torch.abs(torch.sum(gt_plane_points * pred_normals[b, i].unsqueeze(0), dim=1) - 
                                                         pred_distances[b, i])
                                
                                # Mean distance as chamfer component
                                chamfer_component = torch.mean(point_to_plane)
                            else:
                                chamfer_component = torch.tensor(1.0, device=device)
                            
                            # Combined cost (same weights as PVD)
                            cost_matrix[i, j] = normal_sim + 0.5 * dist_diff + 5.0 * chamfer_component

                    # Hungarian matching
                    if num_gt > 0:
                        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
                        pred_indices = torch.tensor(pred_indices, device=device)
                        gt_indices = torch.tensor(gt_indices, device=device)
                        
                        # Create GT classification target based on matching
                        cls_target = torch.zeros_like(pred_logits[b])
                        if len(pred_indices) > 0:
                            cls_target[pred_indices] = 1.0
                        
                        # Classification loss
                        cls_loss += F.binary_cross_entropy_with_logits(pred_logits[b], cls_target)
                        
                        # Parameter loss for matched planes
                        param_normal_loss = 0
                        param_dist_loss = 0
                        this_chamfer_loss = 0
                        
                        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                            # Normal loss (1 - cos similarity)
                            normal_loss = 1 - torch.abs(torch.sum(pred_normals[b, pred_idx] * gt_normals[b, gt_idx]))
                            param_normal_loss += normal_loss
                            
                            # Distance loss
                            dist_loss = torch.abs(pred_distances[b, pred_idx] - gt_distances[b, gt_idx])
                            param_dist_loss += dist_loss
                            
                            # Chamfer loss for points
                            if torch.any(gt_masks[b, gt_idx]):
                                # Get points belonging to this plane
                                gt_plane_points = points[b][gt_masks[b, gt_idx]]
                                
                                # Compute point-to-plane distance for predicted plane
                                point_to_plane = torch.abs(torch.sum(gt_plane_points * pred_normals[b, pred_idx].unsqueeze(0), dim=1) - 
                                                          pred_distances[b, pred_idx])
                                
                                # Mean distance as chamfer loss
                                this_chamfer_loss += torch.mean(point_to_plane)
                        
                        # Average losses by number of matched planes
                        if len(pred_indices) > 0:
                            param_normal_loss /= len(pred_indices)
                            param_dist_loss /= len(pred_indices)
                            this_chamfer_loss /= len(pred_indices)
                        
                        # Add to total parameter and chamfer losses
                        param_loss += param_normal_loss + param_dist_loss
                        pvd_chamfer_loss += this_chamfer_loss

                # Average losses by batch size
                cls_loss /= batch_size
                param_loss /= batch_size
                pvd_chamfer_loss /= batch_size
                
                losses['cls_loss'] = cls_loss
                losses['param_loss'] = param_loss
                losses['pvd_chamfer_loss'] = pvd_chamfer_loss
        
        # === PACO-style Repulsion Loss ===
        repulsion_loss = 0
        if self.repulsion is not None:
            for batch_idx in range(batch_size):
                batch_repulsion = self.compute_repulsion_loss(reconstructed_points, batch_idx)
                repulsion_loss += batch_repulsion
            repulsion_loss /= batch_size
            losses['repulsion_loss'] = repulsion_loss
        
        # === PACO-style Fine-Grained Chamfer Losses ===
        chamfer_norm1_loss = 0
        chamfer_norm2_loss = 0
        
        if gt is not None and gt_index is not None:
            # Process each batch for fine-grained chamfer losses
            for batch_idx in range(batch_size):
                # Get reconstructed points for this batch
                start_indices = torch.arange(self.num_queries, device=device) * self.factor
                end_indices = start_indices + self.factor
                batch_reconstructed_points = torch.stack([
                    reconstructed_points[batch_idx, start:end].reshape(-1, 3)
                    for start, end in zip(start_indices, end_indices)
                ])  # (num_queries, factor, 3)
                
                # Reshape for chamfer computation
                batch_reconstructed_points = batch_reconstructed_points.reshape(1, -1, 3)
                
                # Compute fine-grained chamfer losses
                batch_chamfer_1 = self.compute_fine_chamfer_loss(
                    batch_reconstructed_points, gt[batch_idx], norm=1
                )
                batch_chamfer_2 = self.compute_fine_chamfer_loss(
                    batch_reconstructed_points, gt[batch_idx], norm=2
                )
                
                chamfer_norm1_loss += batch_chamfer_1
                chamfer_norm2_loss += batch_chamfer_2
            
            chamfer_norm1_loss /= batch_size
            chamfer_norm2_loss /= batch_size
            losses['chamfer_norm1_loss'] = chamfer_norm1_loss
            losses['chamfer_norm2_loss'] = chamfer_norm2_loss
        
        # === Additional Confidence Loss ===
        confidence_loss = torch.tensor(0.0, device=device)
        if 'conf' in proxy_outputs and gt_labels is not None:
            # Confidence should be high when prediction is correct
            confidence_target = gt_labels.float()
            confidence_loss = F.mse_loss(proxy_outputs['conf'], confidence_target)
            losses['confidence_loss'] = confidence_loss
        
        # === Total Loss Combination ===
        total_loss = 0
        
        # PVD proxy decoder losses
        if 'cls_loss' in losses:
            total_loss += self.w_classification * losses['cls_loss']
        if 'param_loss' in losses:
            total_loss += self.alpha_param * losses['param_loss']
        if 'pvd_chamfer_loss' in losses:
            total_loss += self.beta_chamfer * losses['pvd_chamfer_loss']
        
        # PACO-style losses
        if 'repulsion_loss' in losses:
            total_loss += self.w_repulsion * losses['repulsion_loss']
        if 'chamfer_norm1_loss' in losses:
            total_loss += self.w_chamfer_norm1 * losses['chamfer_norm1_loss']
        if 'chamfer_norm2_loss' in losses:
            total_loss += self.w_chamfer_norm2 * losses['chamfer_norm2_loss']
        
        # Confidence loss
        if confidence_loss > 0:
            total_loss += self.w_confidence * confidence_loss
        
        losses['total_loss'] = total_loss
        return total_loss, losses


# Additional helper functions for PACO compatibility
class Fold(nn.Module):
    """
    Folding module to generate point clouds from latent features (from PACO)
    """
    
    def __init__(self, in_channel, step, hidden_dim=512, freedom=2):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.freedom = freedom
        
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.register_buffer('folding_seed', torch.cat([a, b], dim=0))
        
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
    Simple Fully Connected Layer for Rebuilding Point Clouds (from PACO)
    """
    
    def __init__(self, input_dims, step, hidden_dim=512, freedom=2):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.freedom = freedom
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, step * freedom)
        )
    
    def forward(self, rec_feature):
        batch_size = rec_feature.size(0)
        rebuild_pc = self.layer(rec_feature).reshape(batch_size, -1, self.step, self.freedom)
        return rebuild_pc


# Additional helper functions for compatibility
def compute_plane_parameters_from_normal_distance(normals, distances):
    """
    Convert normal vectors and distances to plane parameters
    
    Args:
        normals: (B, N, 3) normal vectors
        distances: (B, N) distances from origin
        
    Returns:
        planes: (B, N, 4) plane parameters [a, b, c, d] where ax + by + cz + d = 0
    """
    batch_size, num_planes, _ = normals.shape
    planes = torch.zeros(batch_size, num_planes, 4, device=normals.device)
    
    planes[:, :, :3] = normals  # a, b, c
    planes[:, :, 3] = -distances  # d = -distance
    
    return planes