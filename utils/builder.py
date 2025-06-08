import os

import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler

from .data_utils import build_dataset_from_cfg
from .logger import print_log
from .misc import build_lambda_sche, GradualWarmupScheduler, build_lambda_bnsche, worker_init_fn


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg: Model configuration
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    from paco import MODELS

    return MODELS.build(cfg, **kwargs)


def dataset_builder(args, config, logger=None):
    """Build dataset and dataloader from configuration.
    
    Args:
        args: Command line arguments containing distributed training info
        config: Dataset configuration with parameters like batch size
        
    Returns:
        tuple: (sampler, dataloader) for the specified dataset
    """
    dataset = build_dataset_from_cfg(config)
    shuffle = config.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs if shuffle else 1,
                                                 num_workers=int(args.num_workers),
                                                 drop_last=config.subset == 'train',
                                                 worker_init_fn=worker_init_fn,
                                                 sampler=sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs if shuffle else 1,
                                                 shuffle=shuffle,
                                                 drop_last=config.subset == 'train',
                                                 num_workers=int(args.num_workers),
                                                 worker_init_fn=worker_init_fn)
    if logger is not None:
        print_log(f'[DATASET] {config.subset} set with {len(dataset)} samples, batch size: {config.bs}, shuffle: {shuffle}', logger=logger)
    return sampler, dataloader


def model_builder(config):
    """
    Enhanced model builder to support both PACO and PaCoVoxel models
    """
    model_name = config.name
    
    if model_name == 'PaCoVoxel':
        # Import PaCoVoxel and its dependencies
        try:
            from paco.models.paco_voxel_model import PaCoVoxel
            model = PaCoVoxel(config)
        except ImportError as e:
            raise ImportError(f"Failed to import PaCoVoxel: {e}")
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: PaCoVoxel, PaCo")
    
    return model


def build_optimizer(base_model, config):
    """Build optimizer for the model based on configuration.
    
    Args:
        base_model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            """Add weight decay to specific parameters.
            
            Excludes bias and 1D parameters from weight decay.
            
            Args:
                model: Model to apply weight decay
                weight_decay: Weight decay factor
                skip_list: List of parameter names to skip
                
            Returns:
                list: Parameter groups with appropriate weight decay settings
            """
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer


def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    """Build learning rate scheduler based on configuration.
    
    Args:
        base_model: Model being trained
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        last_epoch: Last epoch number for resuming training
        
    Returns:
        object: Learning rate scheduler or list of schedulers
    """
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.t_max,
                                      lr_min=sche_config.kwargs.min_lr,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      t_in_epochs=True)
    else:
        raise NotImplementedError()

    # Add batch norm momentum scheduler if specified
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)
        scheduler = [scheduler, bnscheduler]

    return scheduler


def resume_model(model, config, logger=None):
    """
    Enhanced model resuming with better error handling for PaCoVoxel
    """
    checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'ckpt-last.pth')
    
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.warning(f'No checkpoint found at {checkpoint_path}')
        return 0, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify model compatibility
        checkpoint_model_name = checkpoint.get('config_model_name', 'Unknown')
        current_model_name = config.model.name
        
        if checkpoint_model_name != current_model_name:
            if logger:
                logger.warning(f'Model name mismatch: checkpoint={checkpoint_model_name}, current={current_model_name}')
        
        # Handle distributed vs non-distributed models
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        epoch = checkpoint.get('epoch', 0)
        best_metrics = checkpoint.get('best_metrics', None)
        
        if logger:
            logger.info(f'Resumed from checkpoint: {checkpoint_path}')
            logger.info(f'Model: {checkpoint_model_name}, Epoch: {epoch}, Best metrics: {best_metrics}')
        
        return epoch, best_metrics
        
    except Exception as e:
        if logger:
            logger.error(f'Error loading checkpoint: {e}')
        return 0, None


def resume_optimizer(optimizer, config, logger=None):
    """
    Enhanced optimizer resuming
    """
    checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'ckpt-last.pth')
    
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.warning(f'No checkpoint found for optimizer at {checkpoint_path}')
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if logger:
                logger.info('Resumed optimizer state')
        else:
            if logger:
                logger.warning('No optimizer state found in checkpoint')
                
    except Exception as e:
        if logger:
            logger.error(f'Error loading optimizer state: {e}')


def save_checkpoint(model, optimizer, epoch, metrics, best_metrics, prefix, config, logger=None):
    """
    Enhanced checkpoint saving with better compatibility for PaCoVoxel
    """
    # Handle distributed vs non-distributed models
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        model_name = getattr(model.module, '__class__').__name__
    else:
        model_state_dict = model.state_dict()
        model_name = model.__class__.__name__
    
    # Save comprehensive checkpoint information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics if metrics is not None else {},
        'best_metrics': best_metrics,
        'config': dict(config) if hasattr(config, 'keys') else config,  # Handle OmegaConf
        'model_name': model_name,
        'config_model_name': config.model.name,  # Store config model name
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if logger:
        logger.info(f'Saved checkpoint: {checkpoint_path}')
        logger.info(f'Model: {model_name}, Epoch: {epoch}, Metrics: {metrics}')
    
    return checkpoint_path


def load_model(model, checkpoint_path, logger=None):
    """
    Enhanced model loading from specific checkpoint with PaCoVoxel support
    """
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.error(f'Checkpoint not found: {checkpoint_path}')
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Log model info if available
        if 'config_model_name' in checkpoint and logger:
            logger.info(f'Loading {checkpoint["config_model_name"]} model from checkpoint')
        
        # Load state dict with error handling
        if hasattr(model, 'module'):
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if logger:
            logger.info(f'Loaded model from: {checkpoint_path}')
            if missing_keys:
                logger.warning(f'Missing keys: {missing_keys}')
            if unexpected_keys:
                logger.warning(f'Unexpected keys: {unexpected_keys}')
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f'Error loading model: {e}')
        return False
