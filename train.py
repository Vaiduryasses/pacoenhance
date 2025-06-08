import os

import torch
from tensorboardX import SummaryWriter
import hydra
from omegaconf import DictConfig

from utils import dist_utils, misc
from utils.runner import run_trainer
from utils.logger import get_root_logger
from utils.config import create_experiment_dir


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """Main function to initialize and run the training process."""

    # Check if CUDA is available and enable cuDNN benchmark for performance
    if cfg.use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

        # Initialize distributed training if applicable
        if cfg.distributed:
            dist_utils.init_dist(cfg.launcher)

            # Re-set GPU IDs when using distributed training
            _, world_size = dist_utils.get_dist_info()
            cfg.world_size = world_size
    
    # Retrieve local rank with default value for non-distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    create_experiment_dir(cfg)

    # Set up logger
    logger = get_root_logger(name=cfg.log_name)

    # Initialize TensorBoard writers for training and validation
    # Only the main process (local_rank == 0) writes to TensorBoard
    if local_rank == 0:
        train_writer = SummaryWriter(os.path.join(cfg.output_dir, 'tensorboard/train'))
        val_writer = SummaryWriter(os.path.join(cfg.output_dir, 'tensorboard/test'))
    else:
        train_writer = None
        val_writer = None

    # Adjust batch size based on the distributed training setting
    if cfg.distributed:
        assert cfg.total_bs % world_size == 0, "Total batch size must be divisible by world size."
        cfg.dataset.bs = cfg.total_bs // world_size
    else:
        cfg.dataset.bs = cfg.total_bs

    # Log distributed training status and model configuration
    if local_rank == 0:
        logger.info(f'Distributed training: {cfg.distributed}')
        
        # Enhanced logging for PaCoVoxel model
        if hasattr(cfg.model, 'name') and cfg.model.name == 'PaCoVoxel':
            logger.info('=== PaCoVoxel Model Configuration ===')
            logger.info(f'Model: {cfg.model.name}')
            logger.info(f'Number of queries: {cfg.model.num_queries}')
            logger.info(f'Number of points: {getattr(cfg.model, "num_points", "Not specified")}')
            logger.info(f'Decoder type: {cfg.model.decoder_type}')
            logger.info(f'Voxel resolution: {getattr(cfg.model, "voxel_resolution", "64")}')
            logger.info(f'Encoder embed dim: {cfg.model.encoder.embed_dim}')
            logger.info(f'Decoder embed dim: {cfg.model.decoder.embed_dim}')
            
            # Log loss weights
            logger.info('Loss weights:')
            logger.info(f'  alpha_param: {getattr(cfg.model, "alpha_param", 0.5)}')
            logger.info(f'  beta_chamfer: {getattr(cfg.model, "beta_chamfer", 20.0)}')
            logger.info(f'  w_classification: {getattr(cfg.model, "w_classification", 1.0)}')
            logger.info(f'  w_repulsion: {getattr(cfg.model, "w_repulsion", 1.0)}')
            logger.info('=====================================')

    # Set random seed for reproducibility if provided
    if cfg.seed is not None:
        if local_rank == 0:
            logger.info(f'Set random seed to {cfg.seed}, deterministic: {cfg.deterministic}')
        misc.set_random_seed(cfg.seed + local_rank, deterministic=cfg.deterministic)

    # In distributed mode, confirm local rank matches the distributed rank
    if cfg.distributed:
        assert local_rank == torch.distributed.get_rank(), "Local rank does not match distributed rank."

    # Enhanced configuration validation for PaCoVoxel
    if hasattr(cfg.model, 'name') and cfg.model.name == 'PaCoVoxel':
        # 验证必需参数
        required_params = {
            'num_queries': 'Number of queries for model',
            'decoder_type': 'Decoder type (fold or fc)',
            'encoder': 'Encoder configuration',
            'decoder': 'Decoder configuration'
        }
        
        for param, description in required_params.items():
            if not hasattr(cfg.model, param):
                logger.error(f'Missing required parameter for PaCoVoxel: {param} ({description})')
                raise ValueError(f'Missing required parameter: {param}')
        
        # 验证encoder和decoder配置
        if not hasattr(cfg.model.encoder, 'embed_dim'):
            logger.error('Missing encoder.embed_dim in configuration')
            raise ValueError('Missing encoder.embed_dim')
        
        if not hasattr(cfg.model.decoder, 'embed_dim'):
            logger.error('Missing decoder.embed_dim in configuration')
            raise ValueError('Missing decoder.embed_dim')
        
        # 确保维度匹配
        if cfg.model.encoder.embed_dim != cfg.model.decoder.embed_dim:
            logger.warning(f'Encoder and decoder embed_dim mismatch: {cfg.model.encoder.embed_dim} vs {cfg.model.decoder.embed_dim}')
        
        # 设置默认损失权重（如果缺失）
        default_weights = {
            'alpha_param': 0.5,
            'beta_chamfer': 20.0,
            'w_classification': 1.0,
            'w_confidence': 0.5,
            'w_repulsion': 1.0,
            'w_chamfer_norm1': 1.0,
            'w_chamfer_norm2': 1.0
        }
        
        for weight_name, default_value in default_weights.items():
            if not hasattr(cfg.model, weight_name):
                setattr(cfg.model, weight_name, default_value)
                logger.info(f'Set default {weight_name} = {default_value}')
        
        if local_rank == 0:
            logger.info('PaCoVoxel configuration validation passed')

    # Run trainer
    try:
        run_trainer(cfg, train_writer, val_writer)
    except Exception as e:
        logger.error(f'Training failed with error: {e}')
        raise
    finally:
        # Ensure writers are closed
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()


if __name__ == '__main__':
    train()