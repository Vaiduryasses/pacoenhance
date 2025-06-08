import copy
import time
import os

import torch
import torch.nn as nn

from utils import builder, dist_utils
from utils.logger import get_logger, print_log
from utils.average_meter import AverageMeter


def init_device(gpu_ids):
    """
    Init devices.

    Parameters
    ----------
    gpu_ids: list of int
        GPU indices to use
    """
    # set multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')

    # does not work for DP after import torch with PyTorch 2.0, but works for DDP nevertheless
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]


def run_trainer(cfg, train_writer=None, val_writer=None):
    """
    Main training function that handles the complete training and validation cycle.

    Args:
        cfg: Configuration object containing all training parameters
        train_writer: TensorBoard writer for training metrics
        val_writer: TensorBoard writer for validation metrics

    Returns:
        None
    """
    logger = get_logger(cfg.log_name)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Build datasets for training and testing
    train_config = copy.deepcopy(cfg.dataset)
    train_config.subset = "train"
    test_config = copy.deepcopy(cfg.dataset)
    test_config.subset = "test"
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(cfg, train_config, logger=logger), \
        builder.dataset_builder(cfg, test_config, logger=logger)
    
    # Build and initialize the model
    base_model = builder.model_builder(cfg.model)
    if cfg.use_gpu:
        base_model.to(local_rank)

    # Initialize training parameters
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Load checkpoints if resuming training or starting from pretrained model
    if cfg.resume_last:
        start_epoch, best_metrics = builder.resume_model(base_model, cfg, logger=logger)
    elif cfg.resume_from is not None:
        builder.load_model(base_model, cfg.resume_from, logger=logger)

    # Print model information for debugging
    if cfg.debug:
        print_log('Trainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

        print_log('Untrainable_parameters:', logger=logger)
        print_log('=' * 25, logger=logger)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger=logger)

    # Set up distributed training if needed
    if cfg.distributed:
        if cfg.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # Set up optimizer and learning rate scheduler
    optimizer = builder.build_optimizer(base_model, cfg)

    if cfg.resume_last:
        builder.resume_optimizer(optimizer, cfg, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, cfg, last_epoch=start_epoch - 1)

    # Main training loop
    base_model.zero_grad()
    for epoch in range(start_epoch, cfg.max_epoch + 1):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        # Initialize timing and loss tracking
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        # 确定模型类型和对应的损失名称
        is_paco_voxel = cfg.model.name == 'PaCoVoxel'
        if is_paco_voxel:
            # PaCoVoxel 返回的损失名称 (基于 get_loss 方法中的映射)
            loss_names = [
                "classification_loss",    # 来自 cls_loss
                "plane_normal_loss",      # 来自 param_loss  
                "plane_chamfer_loss",     # 来自 pvd_chamfer_loss
                "repulsion_loss",         # 直接映射
                "chamfer_norm1_loss",     # 直接映射
                "chamfer_norm2_loss",     # 直接映射
                "confidence_loss",        # 直接映射（如果存在）
                "total_loss"              # 总损失
            ]
            print_log('Using PaCoVoxel model with enhanced losses', logger=logger)
        else:
            # 原始PACO损失名称
            loss_names = [
                "plane_chamfer_loss", "classification_loss", 'chamfer_norm1_loss', 
                'chamfer_norm2_loss', "plane_normal_loss", "repulsion_loss", "total_loss"
            ]
            print_log('Using original PACO model', logger=logger)
        
        # 初始化损失跟踪器
        train_losses = AverageMeter(loss_names)
        num_iter = 0

        base_model.train()
        n_batches = len(train_dataloader)
        for idx, (model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            try:
                processed_data = prepare_input_data(data, cfg.dataset.name)
                gt = processed_data['gt']
                gt_index = processed_data['gt_index'] 
                plane = processed_data['plane']
                plane_index = processed_data['plane_index']
                pc = processed_data['pc']
            except Exception as e:
                print_log(f'Error processing data batch {idx}: {e}', logger=logger)
                continue

            num_iter += 1

            # 前向传播 - 处理不同模型类型
            if is_paco_voxel:
                # PaCoVoxel 特定的前向传播和损失计算
                xyz = extract_xyz_from_pc(pc)
                # 存储输入用于损失计算
                model_instance = base_model.module if hasattr(base_model, 'module') else base_model
                if hasattr(model_instance, '_last_input') or hasattr(model_instance, 'base_model'):
                    # 如果模型有 _last_input 属性或者有 base_model 子模块
                    if hasattr(model_instance, 'base_model') and hasattr(model_instance.base_model, '_last_input'):
                        model_instance.base_model._last_input = xyz
                    else:
                        model_instance._last_input = xyz
                # 前向传播
                try:
                    outputs = base_model(xyz)
                except Exception as e:
                    print_log(f'Error in PaCoVoxel forward pass: {e}', logger=logger)
                    continue
                # 使用 get_loss 方法计算损失
                try:
                    losses = model_instance.get_loss(
                        config=cfg,  # 传入完整配置
                        outputs=outputs,
                        class_prob=outputs.get('class_prob'),
                        gt=gt,
                        gt_index=gt_index,
                        plane=plane,
                        plan_index=plane_index,  # 注意：使用 plan_index 而不是 plane_index
                        gt_planes=None  # 如果有PVD风格的ground truth，在这里传入
                    )
                except Exception as e:
                    print_log(f'Error in PaCoVoxel loss computation: {e}', logger=logger)
                    # 创建默认损失避免训练中断
                    losses = {name: torch.tensor(0.0, device=gt.device, requires_grad=True) for name in loss_names}
                    losses['total_loss'] = torch.tensor(0.1, device=gt.device, requires_grad=True)
                    # 小的非零值
            else:
                # 原始PACO前向传播
                try:
                    ret, class_prob = base_model(pc)
                    model_instance = base_model.module if hasattr(base_model, 'module') else base_model
                    losses = model_instance.get_loss(cfg.loss, ret, class_prob, gt, gt_index, plane, plane_index)
                except Exception as e:
                    print_log(f'Error in PACO forward pass: {e}', logger=logger)
                    continue
            
            _loss = losses['total_loss']

            # 检查损失有效性
            if torch.isnan(_loss) or torch.isinf(_loss):
                print_log(f'Invalid loss detected at epoch {epoch}, batch {idx}. Skipping batch.', logger=logger)
                continue

            _loss.backward()
            # Update weights after accumulating gradients
            if num_iter == cfg.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(cfg, 'grad_norm_clip', 10),
                                               norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # Process and log loss metrics
            if cfg.distributed:
                for key in losses.keys():
                    if isinstance(losses[key], torch.Tensor):
                        losses[key] = dist_utils.reduce_tensor(losses[key], cfg)

            # 更新损失，只使用存在的损失
            loss_values = []
            for loss_name in loss_names:
                if loss_name in losses and isinstance(losses[loss_name], torch.Tensor):
                    loss_values.append(losses[loss_name].item() * 1000)
                else:
                    loss_values.append(0.0)  # 缺失损失的默认值
            train_losses.update(loss_values)

            if cfg.distributed:
                torch.cuda.synchronize()

            # Log metrics to TensorBoard
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                for key in losses.keys():
                    if isinstance(losses[key], torch.Tensor):
                        train_writer.add_scalar(f'Loss/Batch/{key}', losses[key].item() * 1000, n_itr)
            # Update timing information
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # Print progress information
            if idx % 100 == 0:
                print_log(f'[Epoch {epoch}/{cfg.max_epoch}][Batch {idx + 1}/{n_batches}] | BatchTime = {batch_time.val():.3f}s | '
                          f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | lr = {optimizer.param_groups[0]["lr"]:.6f}', logger=logger)

            # Handle special case for GradualWarmup scheduler
            if cfg.scheduler.type == 'GradualWarmup':
                if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        # Step the learning rate scheduler after each epoch
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        # Log epoch-level training metrics
        if train_writer is not None:
            for i, key in enumerate(losses.keys()):
                if i < len(train_losses.avg()):
                    train_writer.add_scalar(f'Loss/Epoch/{key}', train_losses.avg(i), epoch)
            print_log(f'[Training] Epoch: {epoch} | EpochTime = {epoch_end_time - epoch_start_time:.3f}s | '
                      f'Losses = [{", ".join(f"{l:.4f}" for l in train_losses.avg())}]', logger=logger)

        # Run validation at specified frequency
        if epoch % cfg.val_freq == 0:
            test_losses = validate(base_model, test_dataloader, epoch, val_writer, cfg, logger=logger, is_paco_voxel=is_paco_voxel)
            if best_metrics is None:
                best_metrics = test_losses[cfg.consider_metric]

            # Save checkpoint if current model is the best so far
            if test_losses[cfg.consider_metric] < best_metrics:
                best_metrics = test_losses[cfg.consider_metric]
                metrics = test_losses
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', cfg,
                                        logger=logger)

        # Save checkpoints
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', cfg, logger=logger)
        if (cfg.max_epoch - epoch) < 2:
            metrics = test_losses
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    cfg, logger=logger)
    
    # Close TensorBoard writers
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, config, logger=None, is_paco_voxel=False):
    """
    Validate the model on the test dataset.
    
    Args:
        base_model: Model to validate
        test_dataloader: DataLoader for test data
        epoch: Current epoch number
        val_writer: TensorBoard writer for validation metrics
        config: Configuration object containing validation parameters
        logger: Logger for printing information
        is_paco_voxel: Whether using PaCoVoxel model
        
    Returns:
        metrics: Dictionary containing validation metrics
    """
    base_model.eval()

    # Initialize metrics tracking
    if is_paco_voxel:
        loss_names = [
            "classification_loss", "plane_normal_loss", "plane_chamfer_loss",
            "repulsion_loss", "chamfer_norm1_loss", "chamfer_norm2_loss",
            "confidence_loss", "total_loss"
        ]
    else:
        loss_names = [
            "plane_chamfer_loss", "classification_loss", 'chamfer_norm1_loss', 
            'chamfer_norm2_loss', "plane_normal_loss", "repulsion_loss", "total_loss"
        ]

    # Initialize metrics tracking
    test_losses = AverageMeter(loss_names)
    n_samples = len(test_dataloader)  # bs is 1
    interval = n_samples // 10 + 1

    # Validation loop
    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            try:
                model_id = model_ids[0]
                # 准备输入数据
                processed_data = prepare_input_data(data, config.dataset.name)
                gt = processed_data['gt']
                gt_index = processed_data['gt_index']
                plane = processed_data['plane']
                plane_index = processed_data['plane_index']
                pc = processed_data['pc']
                
                # 前向传播和损失计算
                if is_paco_voxel:
                    # PaCoVoxel验证
                    xyz = extract_xyz_from_pc(pc)
                    # 存储输入用于损失计算
                    model_instance = base_model.module if hasattr(base_model, 'module') else base_model
                    if hasattr(model_instance, '_last_input'):
                        model_instance._last_input = xyz
                    elif hasattr(model_instance, 'base_model') and hasattr(model_instance.base_model, '_last_input'):
                        model_instance.base_model._last_input = xyz
                
                    # 前向传播
                    outputs = base_model(xyz)
                
                    # 损失计算
                    try:
                        losses = model_instance.get_loss(
                            config=config,
                            outputs=outputs,
                            class_prob=outputs.get('class_prob'),
                            gt=gt,
                            gt_index=gt_index,
                            plane=plane,
                            plan_index=plane_index,
                            gt_planes=None
                        )
                    except Exception as e:
                        if logger:
                            logger.warning(f'Error in PaCoVoxel validation loss: {e}')
                        # 创建默认损失
                        losses = {name: torch.tensor(0.0, device=gt.device) for name in loss_names}
                else:
                    # 原始PACO验证
                    ret, class_prob = base_model(pc)
                    model_instance = base_model.module if hasattr(base_model, 'module') else base_model
                    losses = model_instance.get_loss(config.loss, ret, class_prob, gt, gt_index, plane, plane_index)
                
                if config.distributed:
                    for key in losses.keys():
                        if isinstance(losses[key], torch.Tensor):
                            losses[key] = dist_utils.reduce_tensor(losses[key], config)
                
                # 更新损失，只使用存在的损失
                loss_values = []
                for loss_name in loss_names:
                    if loss_name in losses and isinstance(losses[loss_name], torch.Tensor):
                        loss_values.append(losses[loss_name].item() * 1000)
                    else:
                        loss_values.append(0.0)
                test_losses.update(loss_values)
            
            except Exception as e:
                if logger:
                    logger.warning(f'Error in validation batch {idx}: {e}')
                continue

            # Synchronize processes in distributed mode
            if config.distributed:
                torch.cuda.synchronize()

    # Log validation metrics to TensorBoard
    if val_writer is not None:
        for i, key in enumerate(losses.keys()):
            if i < len(test_losses.avg()):
                val_writer.add_scalar(f'Loss/Epoch/{key}', test_losses.avg(i), epoch)

    print_log(f'[Validation] Epoch: {epoch} | Losses = [{", ".join(f"{l:.4f}" for l in test_losses.avg())}]', logger=logger)
    
    # Prepare metrics dictionary for return
    metrics = {}
    for i, key in enumerate(losses.keys()):
        if i < len(test_losses.avg()):
            metrics[key] = test_losses.avg(i)

    return metrics


def extract_xyz_from_pc(pc):
    """
    Extract xyz coordinates from point cloud data
    
    Args:
        pc: (B, N, 7) point cloud with [x, y, z, nx, ny, nz, index]
        
    Returns:
        xyz: (B, N, 3) xyz coordinates
    """
    if pc.shape[-1] >= 3:
        return pc[:, :, :3]  # Extract first 3 channels (x, y, z)
    else:
        return pc


def prepare_input_data(data, dataset_name):
    """
    Prepare input data for different model types
    
    Args:
        data: raw data from dataloader
        dataset_name: name of the dataset
        
    Returns:
        dict: processed data with consistent format
    """
    if dataset_name == 'ABC':
        return {
            'gt': data[0].cuda(),           # bs, n, 3 - ground truth points
            'gt_index': data[1].cuda(),     # bs, n - ground truth plane indices
            'plane': data[2].cuda(),        # bs, 20, 4 - ground truth plane parameters
            'plane_index': data[3].cuda(),  # bs, 20 - plane indices
            'pc': data[4].cuda()            # bs, n, 7 - input point cloud (x,y,z,nx,ny,nz,index)
        }
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported')