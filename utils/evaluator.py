import os
import multiprocessing

import numpy as np
from omegaconf import DictConfig


def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    """
    Start a process pool to execute worker function with given parameters.
    
    Parameters
    ----------
    worker_function: callable
        Function to be executed in parallel
    parameters: list
        List of parameter tuples to pass to the worker function
    num_processes: int
        Number of processes to use
    timeout: int, optional
        Maximum time to wait for processes to complete
        
    Returns
    -------
    list or None
        Results from worker function calls, or None if parameters is empty
    """
    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print('Running loop for {} with {} calls on {} subprocess workers'.format(
            str(worker_function), len(parameters), num_processes))
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None

def _chamfer_distance_single_file(file_in, file_ref, samples_per_model, num_processes=1):
    """
    Calculate chamfer distance between two mesh files.
    
    Parameters
    ----------
    file_in: str
        Path to the input mesh file
    file_ref: str
        Path to the reference mesh file
    samples_per_model: int
        Number of points to sample on each mesh
    num_processes: int, default=1
        Number of processes for KDTree queries
        
    Returns
    -------
    tuple
        (input_file_path, reference_file_path, chamfer_distance)
    """
    import trimesh
    import trimesh.sample
    import sys
    import scipy.spatial as spatial

    def sample_mesh(mesh_file, num_samples):
     """
     Sample points on mesh surface.
    
     Parameters
     ----------
     mesh_file: str
        Path to mesh file
     num_samples: int
        Number of points to sample
        
     Returns
     -------
     tuple
        (samples: numpy.ndarray, error_msg: str)
        samples: shape (num_samples, 3) if successful, (0, 3) if failed
        error_msg: error description if failed, empty string if successful
     """
     import os
    
    # 检查文件是否存在
     if not os.path.exists(mesh_file):
        return np.zeros((0, 3)), f"File not found: {mesh_file}"
    
     try:
        # 加载网格
        mesh = trimesh.load(mesh_file, process=False)
        
        # 检查网格是否有效
        if mesh is None:
            return np.zeros((0, 3)), f"Failed to load mesh from: {mesh_file}"
        
        # 检查网格是否有面
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            return np.zeros((0, 3)), f"Mesh has no faces: {mesh_file}"
        
        # 进行采样
        samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
        return samples, ""
        
     except Exception as e:
        return np.zeros((0, 3)), f"Error processing {mesh_file}: {str(e)}"

    # 使用修改后的函数
    new_mesh_samples, new_error = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples, ref_error = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0:
        print(f"New mesh failed: {new_error}")
    if ref_mesh_samples.shape[0] == 0:
        print(f"Reference mesh failed: {ref_error}")

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 1.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    ref_new_dist, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, workers=num_processes)
    new_ref_dist, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, workers=num_processes)

    ref_new_dist_sum = np.sum(ref_new_dist)
    new_ref_dist_sum = np.sum(new_ref_dist)
    chamfer_dist = (ref_new_dist_sum + new_ref_dist_sum) / samples_per_model

    return file_in, file_ref, chamfer_dist
    
def _hausdorff_distance_single_file(file_in, file_ref, samples_per_model):
    """
    Calculate Hausdorff distance between two mesh files.
    
    Parameters
    ----------
    file_in: str
        Path to the input mesh file
    file_ref: str
        Path to the reference mesh file
    samples_per_model: int
        Number of points to sample on each mesh
        
    Returns
    -------
    tuple
        (input_file_path, reference_file_path, distance_new_to_ref, distance_ref_to_new, max_distance)
    """
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        """
        Sample points on mesh surface.
        
        Parameters
        ----------
        mesh_file: str
            Path to mesh file
        num_samples: int
            Number of points to sample
            
        Returns
        -------
        numpy.ndarray
            Sampled points, shape (num_samples, 3)
        """
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            return samples
        except:
            return np.zeros((0, 3))
        

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 1.0, 1.0, 1.0

    dist_new_ref, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    dist_ref_new, _, _ = spatial.distance.directed_hausdorff(ref_mesh_samples, new_mesh_samples)
    dist = max(dist_new_ref, dist_ref_new)
    
    return file_in, file_ref, dist_new_ref, dist_ref_new, dist

# def _normal_consistency(file_in, file_ref, samples_per_model, num_processes=1):
#     """
#     Normal consistency with simple progress monitoring.
#     """
#     import scipy.spatial as spatial
#     import trimesh
#     import sys
#     import trimesh.sample
#     import time
    
#     print(f"[NC] 开始处理: {file_in}")
#     print(f"[NC] 参考文件: {file_ref}")
#     print(f"[NC] 采样数量: {samples_per_model}")
    
#     def sample_mesh(mesh_file, num_samples):
#         try:
#             mesh = trimesh.load(mesh_file, process=False)
#             samples, sample_face_indices = trimesh.sample.sample_surface(mesh, num_samples)
#             face_normals = np.array(mesh.face_normals)
#             normals = face_normals[sample_face_indices]
#             return samples, normals
#         except:
#             return np.zeros((0, 3)), np.zeros((0, 3))

#     print("[NC] 1/5 采样输入网格...")
#     new_mesh_samples, new_normals = sample_mesh(file_in, samples_per_model)
#     print(f"[NC] 输入网格采样完成: {new_mesh_samples.shape[0]}个点")
    
#     print("[NC] 2/5 采样参考网格...")
#     ref_mesh_samples, ref_normals = sample_mesh(file_ref, samples_per_model)
#     print(f"[NC] 参考网格采样完成: {ref_mesh_samples.shape[0]}个点")

#     if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
#         print("[NC] 采样失败，返回0.0")
#         return file_in, file_ref, 0.0

#     print("[NC] 3/5 构建KDTree...")
#     leaf_size = 100
#     sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
#     kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leafsize=leaf_size)
#     kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leafsize=leaf_size)
#     print("[NC] KDTree构建完成")

#     print("[NC] 4/5 执行KDTree查询...")
#     print("[NC] 查询1: 寻找最近邻...")
#     _, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, k=1, workers=1)
#     print("[NC] 查询2: 寻找最近邻...")
#     _, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, k=1, workers=1)
#     print("[NC] KDTree查询完成")

#     print("[NC] 5/5 计算法向量一致性...")
#     normals_dot_pred_gt = (np.abs(np.sum(new_normals * ref_normals[corr_ref_ids], axis=1)).mean())
#     normals_dot_gt_pred = (np.abs(np.sum(ref_normals * new_normals[corr_new_ids], axis=1)).mean())
#     normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

#     print(f"[NC] 完成! 法向量一致性: {normal_consistency:.6f}")
#     return file_in, file_ref, normal_consistency
def _normal_consistency_gpu(file_in, file_ref, samples_per_model, device='cuda'):
    """
    GPU加速的法向量一致性计算
    
    注意：函数签名必须与调用参数匹配：
    (file_in, file_ref, samples_per_model, device)
    """
    import numpy as np
    import trimesh
    import trimesh.sample
    
    # 检查GPU可用性
    try:
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"CUDA不可用，回退到CPU计算文件: {file_in}")
            device = 'cpu'
        # print(f"处理文件: {file_in} -> {file_ref}, 设备: {device}")
    except ImportError:
        print(f"PyTorch未安装，回退到CPU计算文件: {file_in}")
        return _normal_consistency_cpu_fallback(file_in, file_ref, samples_per_model)
    
    def sample_mesh(mesh_file, num_samples):
        """采样网格表面点和法向量"""
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, sample_face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            face_normals = np.array(mesh.face_normals)
            normals = face_normals[sample_face_indices]
            return samples.astype(np.float32), normals.astype(np.float32)
        except Exception as e:
            print(f"网格采样失败 {mesh_file}: {e}")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # 采样网格
    new_mesh_samples, new_normals = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples, ref_normals = sample_mesh(file_ref, samples_per_model)
    
    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 0.0

    # 如果数据太小，直接用CPU
    if new_mesh_samples.shape[0] < 1000 or ref_mesh_samples.shape[0] < 1000:
        return _normal_consistency_cpu_fallback(file_in, file_ref, samples_per_model)

    try:
        # 转换为PyTorch tensor
        new_samples_gpu = torch.from_numpy(new_mesh_samples).to(device)
        new_normals_gpu = torch.from_numpy(new_normals).to(device)
        ref_samples_gpu = torch.from_numpy(ref_mesh_samples).to(device)
        ref_normals_gpu = torch.from_numpy(ref_normals).to(device)
        
        # GPU计算
        with torch.no_grad():
            # 分块计算以避免内存问题
            if new_mesh_samples.shape[0] > 5000 or ref_mesh_samples.shape[0] > 5000:
                normal_consistency = gpu_normal_consistency_chunked(
                    new_samples_gpu, new_normals_gpu,
                    ref_samples_gpu, ref_normals_gpu,
                    device, chunk_size=2000
                )
            else:
                normal_consistency = gpu_normal_consistency_optimized(
                    new_samples_gpu, new_normals_gpu,
                    ref_samples_gpu, ref_normals_gpu,
                    device
                )
            
            # 清理GPU内存
            del new_samples_gpu, new_normals_gpu, ref_samples_gpu, ref_normals_gpu
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            return file_in, file_ref, normal_consistency
            
    except Exception as e:
        print(f"GPU计算失败 {file_in}: {e}, 回退到CPU")
        if device == 'cuda':
            torch.cuda.empty_cache()
        return _normal_consistency_cpu_fallback(file_in, file_ref, samples_per_model)


def gpu_normal_consistency_optimized(new_samples, new_normals, ref_samples, ref_normals, device):
    """优化的GPU法向量一致性计算"""
    import torch
    
    # 计算最近邻
    distances_ref_to_new = torch.cdist(ref_samples, new_samples)
    corr_ref_ids = torch.argmin(distances_ref_to_new, dim=1)
    
    distances_new_to_ref = torch.cdist(new_samples, ref_samples)
    corr_new_ids = torch.argmin(distances_new_to_ref, dim=1)
    
    # 计算法向量点积
    normals_dot_pred_gt = torch.abs(
        torch.sum(new_normals * ref_normals[corr_ref_ids], dim=1)
    ).mean()
    
    normals_dot_gt_pred = torch.abs(
        torch.sum(ref_normals * new_normals[corr_new_ids], dim=1)
    ).mean()
    
    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2.0
    return normal_consistency.cpu().item()


def gpu_normal_consistency_chunked(new_samples, new_normals, ref_samples, ref_normals, 
                                 device, chunk_size=2000):
    """分块GPU法向量一致性计算"""
    import torch
    
    n_new = new_samples.shape[0]
    n_ref = ref_samples.shape[0]
    
    all_dot_pred_gt = []
    all_dot_gt_pred = []
    
    # 分块处理 ref -> new
    for i in range(0, n_ref, chunk_size):
        end_i = min(i + chunk_size, n_ref)
        ref_chunk = ref_samples[i:end_i]
        ref_normals_chunk = ref_normals[i:end_i]
        
        distances = torch.cdist(ref_chunk, new_samples)
        corr_ids = torch.argmin(distances, dim=1)
        
        dot_products = torch.abs(
            torch.sum(new_normals[corr_ids] * ref_normals_chunk, dim=1)
        )
        all_dot_pred_gt.append(dot_products)
    
    # 分块处理 new -> ref
    for i in range(0, n_new, chunk_size):
        end_i = min(i + chunk_size, n_new)
        new_chunk = new_samples[i:end_i]
        new_normals_chunk = new_normals[i:end_i]
        
        distances = torch.cdist(new_chunk, ref_samples)
        corr_ids = torch.argmin(distances, dim=1)
        
        dot_products = torch.abs(
            torch.sum(ref_normals[corr_ids] * new_normals_chunk, dim=1)
        )
        all_dot_gt_pred.append(dot_products)
    
    # 合并结果
    normals_dot_pred_gt = torch.cat(all_dot_pred_gt).mean()
    normals_dot_gt_pred = torch.cat(all_dot_gt_pred).mean()
    
    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2.0
    return normal_consistency.cpu().item()


def _normal_consistency_cpu_fallback(file_in, file_ref, samples_per_model, num_processes=1):
    """CPU回退版本"""
    import scipy.spatial as spatial
    import trimesh
    import sys
    import trimesh.sample
    import numpy as np
    
    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file, process=False)
            samples, sample_face_indices = trimesh.sample.sample_surface(mesh, num_samples)
            face_normals = np.array(mesh.face_normals)
            normals = face_normals[sample_face_indices]
            return samples, normals
        except:
            return np.zeros((0, 3)), np.zeros((0, 3))

    new_mesh_samples, new_normals = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples, ref_normals = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, 0.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    _, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, k=1, workers=num_processes)
    _, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, k=1, workers=num_processes)
    
    normals_dot_pred_gt = (np.abs(np.sum(new_normals * ref_normals[corr_ref_ids], axis=1)).mean())
    normals_dot_gt_pred = (np.abs(np.sum(ref_normals * new_normals[corr_new_ids], axis=1)).mean())

    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

    return file_in, file_ref, normal_consistency

# def mesh_comparison(new_meshes_dir_abs, ref_meshes_dir_abs,
#                     num_processes, report_name, samples_per_model=10000, dataset_file_abs=None):
#     """
#     Compare meshes in two directories and calculate metrics.
    
#     This function calculates Hausdorff distance, Chamfer distance, and normal consistency
#     between corresponding meshes in the input and reference directories.
    
#     Parameters
#     ----------
#     new_meshes_dir_abs: str
#         Path to directory containing input meshes
#     ref_meshes_dir_abs: str
#         Path to directory containing reference meshes
#     num_processes: int
#         Number of processes to use for parallel computation
#     report_name: str
#         Path to output CSV report file
#     samples_per_model: int, default=10000
#         Number of points to sample on each mesh
#     dataset_file_abs: str, optional
#         Path to file listing specific models to evaluate
        
#     Returns
#     -------
#     list
#         Results of mesh comparisons
#     """
#     if not os.path.isdir(new_meshes_dir_abs):
#         print('Warning: dir to check doesn\'t exist'.format(new_meshes_dir_abs))
#         return

#     new_mesh_files = [f for f in os.listdir(new_meshes_dir_abs)
#                       if os.path.isfile(os.path.join(new_meshes_dir_abs, f))]
#     ref_mesh_files = [f for f in os.listdir(ref_meshes_dir_abs)
#                       if os.path.isfile(os.path.join(ref_meshes_dir_abs, f))]
    
#     if dataset_file_abs is None:
#         mesh_files_to_compare_set = set(ref_mesh_files)  # set for efficient search
#     else:
#         if not os.path.isfile(dataset_file_abs):
#             raise ValueError('File does not exist: {}'.format(dataset_file_abs))
#         with open(dataset_file_abs) as f:
#             mesh_files_to_compare_set = f.readlines()
#             mesh_files_to_compare_set = [f.replace('\n', '') + '.obj' for f in mesh_files_to_compare_set]
#             mesh_files_to_compare_set = set(mesh_files_to_compare_set)
    

#     def ref_mesh_for_new_mesh(new_mesh_file: str, all_ref_meshes: list) -> list:
#         """
#         Find corresponding reference meshes for a given input mesh.
        
#         Parameters
#         ----------
#         new_mesh_file: str
#             Input mesh filename
#         all_ref_meshes: list
#             List of all reference mesh filenames
            
#         Returns
#         -------
#         list
#             Matching reference mesh filenames
#         """
#         stem_new_mesh_file = new_mesh_file.split('.')[0]
#         ref_files = list(set([f for f in all_ref_meshes if f.split('.')[0] == stem_new_mesh_file]))
#         return ref_files

#     call_params = []
#     for fi, new_mesh_file in enumerate(new_mesh_files):
#         if new_mesh_file in mesh_files_to_compare_set:
#             new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
#             ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
#             if len(ref_mesh_files_matching) > 0:
#                 ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
#                 call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
#     if len(call_params) == 0:
#         raise ValueError('Results are empty!')
#     results_hausdorff = start_process_pool(_hausdorff_distance_single_file, call_params, num_processes)
#     results = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in results_hausdorff]

#     call_params = []
   
#     for fi, new_mesh_file in enumerate(new_mesh_files):
#         if new_mesh_file in mesh_files_to_compare_set:
#             new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
#             ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
#             if len(ref_mesh_files_matching) > 0:
#                 ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
#                 call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 1))
#     results_chamfer = start_process_pool(_chamfer_distance_single_file, call_params, num_processes)
#     results = [r + (str(results_chamfer[ri][2]),) for ri, r in enumerate(results)]
    
#     call_params = []
#     for fi, new_mesh_file in enumerate(new_mesh_files):
#         if new_mesh_file in mesh_files_to_compare_set:
#             new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
#             ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
#             if len(ref_mesh_files_matching) > 0:
#                  ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
#                  call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 1))
#     call_params_gpu = [(new_file, ref_file, samples_per_model, 'cuda') for new_file, ref_file, samples_per_model in call_params]
#     results_normal = start_process_pool(_normal_consistency_gpu, call_params_gpu, 1)
#     results = [r + (str(results_normal[ri][2]),) for ri, r in enumerate(results)]
    
#     # filter out failed results
#     failed_results = [r for r in results if r[-1] == "0.0"]
#     results = [r for r in results if r[-1] != "0.0"]

    
#     for fi, new_mesh_file in enumerate(new_mesh_files):
#         if new_mesh_file not in mesh_files_to_compare_set:
#             if dataset_file_abs is None:
#                 new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
#                 ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
#                 if len(ref_mesh_files_matching) > 0:
#                     reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
#                     failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(2), str(2), str(2), str(2), str(0)))
#         else:
#             mesh_files_to_compare_set.remove(new_mesh_file)
            
#     # no reconstruction but reference
#     for ref_without_new_mesh in mesh_files_to_compare_set:
#         new_mesh_file_abs = os.path.join(new_meshes_dir_abs, ref_without_new_mesh)
#         reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_without_new_mesh)
#         failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(1), str(1), str(1), str(1), str(0)))

#     # sort by file name
#     failed_results = sorted(failed_results, key=lambda x: x[0])
#     results = sorted(results, key=lambda x: x[0])
#     with open(report_name, 'w') as f:
#         f.write('in mesh,ref mesh,Hausdorff dist new-ref,Hausdorff dist ref-new,Hausdorff dist,Chamfer dist(1: no input),Normal consistency(0: no input)\n')
#         for r in failed_results:
#             f.write(','.join(r) + '\n')
#         for r in results:
#             f.write(','.join(r) + '\n')


#     return results
def mesh_comparison(new_meshes_dir_abs, ref_meshes_dir_abs,
                    num_processes, report_name, samples_per_model=10000, dataset_file_abs=None):
    """
    Compare meshes in two directories and calculate metrics.
    """
    if not os.path.isdir(new_meshes_dir_abs):
        print('Warning: dir to check doesn\'t exist'.format(new_meshes_dir_abs))
        return

    new_mesh_files = [f for f in os.listdir(new_meshes_dir_abs)
                      if os.path.isfile(os.path.join(new_meshes_dir_abs, f))]
    ref_mesh_files = [f for f in os.listdir(ref_meshes_dir_abs)
                      if os.path.isfile(os.path.join(ref_meshes_dir_abs, f))]
    
    if dataset_file_abs is None:
        mesh_files_to_compare_set = set(ref_mesh_files)
    else:
        if not os.path.isfile(dataset_file_abs):
            raise ValueError('File does not exist: {}'.format(dataset_file_abs))
        with open(dataset_file_abs) as f:
            mesh_files_to_compare_set = f.readlines()
            mesh_files_to_compare_set = [f.replace('\n', '') + '.obj' for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = set(mesh_files_to_compare_set)

    def ref_mesh_for_new_mesh(new_mesh_file: str, all_ref_meshes: list) -> list:
        stem_new_mesh_file = new_mesh_file.split('.')[0]
        ref_files = list(set([f for f in all_ref_meshes if f.split('.')[0] == stem_new_mesh_file]))
        return ref_files

    # Hausdorff distance calculation
    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    
    if len(call_params) == 0:
        raise ValueError('Results are empty!')
    
    results_hausdorff = start_process_pool(_hausdorff_distance_single_file, call_params, num_processes)
    results = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in results_hausdorff]

    # Chamfer distance calculation
    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 1))
    
    results_chamfer = start_process_pool(_chamfer_distance_single_file, call_params, num_processes)
    results = [r + (str(results_chamfer[ri][2]),) for ri, r in enumerate(results)]
    
    # Normal consistency calculation - GPU版本
    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                # 修复：正确构建GPU版本的参数
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model, 'cuda'))
    
    # 使用GPU版本的法向量一致性计算
    print(f"Running GPU normal consistency calculation with {len(call_params)} pairs...")
    results_normal = start_process_pool(_normal_consistency_gpu, call_params, 1)  # GPU通常用单进程
    results = [r + (str(results_normal[ri][2]),) for ri, r in enumerate(results)]
    
    # Filter out failed results
    failed_results = [r for r in results if r[-1] == "0.0"]
    results = [r for r in results if r[-1] != "0.0"]

    # Handle missing files
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file not in mesh_files_to_compare_set:
            if dataset_file_abs is None:
                new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
                ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
                if len(ref_mesh_files_matching) > 0:
                    reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                    failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(2), str(2), str(2), str(2), str(0)))
        else:
            mesh_files_to_compare_set.remove(new_mesh_file)
            
    # No reconstruction but reference
    for ref_without_new_mesh in mesh_files_to_compare_set:
        new_mesh_file_abs = os.path.join(new_meshes_dir_abs, ref_without_new_mesh)
        reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_without_new_mesh)
        failed_results.append((new_mesh_file_abs, reference_mesh_file_abs, str(1), str(1), str(1), str(1), str(0)))

    # Sort by file name
    failed_results = sorted(failed_results, key=lambda x: x[0])
    results = sorted(results, key=lambda x: x[0])
    
    with open(report_name, 'w') as f:
        f.write('in mesh,ref mesh,Hausdorff dist new-ref,Hausdorff dist ref-new,Hausdorff dist,Chamfer dist(1: no input),Normal consistency(0: no input)\n')
        for r in failed_results:
            f.write(','.join(r) + '\n')
        for r in results:
            f.write(','.join(r) + '\n')

    return results
    
def generate_stats(cfg: DictConfig):
    """
    Evaluate hausdorff distance, chamfer distance and normal consistency between reconstructed and GT models.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration containing:
        - reconstruction_dir: Directory with reconstructed meshes
        - reference_dir: Directory with ground truth meshes
        - num_workers: Number of parallel workers
        - csv_path: Output file path
        - evaluate.num_samples: Number of points to sample per mesh
    """
    
    mesh_comparison(
        new_meshes_dir_abs=cfg.reconstruction_dir,
        ref_meshes_dir_abs=cfg.reference_dir,
        num_processes=cfg.num_workers,
        report_name=cfg.csv_path,
        samples_per_model=cfg.evaluate.num_samples,
        dataset_file_abs=None)


if __name__ == '__main__':
    generate_stats()
