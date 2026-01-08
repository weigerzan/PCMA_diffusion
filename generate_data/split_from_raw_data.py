"""
从原始.dat文件出发，切片读取，判断是否能解调，打标签，然后分训练集测试集

使用方式：python split_from_raw_data.py [--config configs/base_config.yaml]
配置从 YAML 文件的 data_generation.split 部分读取
"""

import json
import numpy as np
from pathlib import Path
import os
import argparse
import yaml
from multiprocessing import Pool, cpu_count
import time

# 导入解调和评估函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import (
    FS, TARGET_SPS, SLICE_LENGTH,
    downsample_to_sps8,
    find_optimal_loop_bandwidth,
    evaluate_clustering_quality,
    rrc_filter
)


# ============= 配置加载函数 =============
def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'data_generation' not in config:
        raise ValueError("YAML 配置文件中缺少 'data_generation' 部分")
    
    raw_data_cfg = config['data_generation'].get('raw_data', {})
    split_cfg = config['data_generation'].get('split', {})
    
    # 构建配置字典
    config_dict = {
        'modulation_list': split_cfg.get('modulation_list', ['8PSK']),
        'output_dir': split_cfg.get('output_dir', '/nas/datasets/yixin/PCMA/real_data'),
        'threshold': split_cfg.get('threshold', 6.0),
        'train_ratio': split_cfg.get('train_ratio', 0.9),
        'random_seed': split_cfg.get('random_seed', 42),
        'max_slices_per_file': split_cfg.get('max_slices_per_file'),
        'max_samples': split_cfg.get('max_samples'),
        'mode': split_cfg.get('mode', 'both'),
        'num_workers': split_cfg.get('num_workers'),
        # 原始数据路径配置
        'raw_data_base_dir': raw_data_cfg.get('base_dir', '/nas/datasets/LYX/PCMA'),
        'raw_data_paths': raw_data_cfg.get('paths', {}),
    }
    
    return config_dict


# ============= 辅助函数 =============
def find_dat_file(modulation, raw_data_base_dir, raw_data_paths):
    """
    根据调制方式查找.dat文件路径。
    优先使用raw_data_paths中的固定路径，如果不存在则尝试在raw_data_base_dir中查找。
    """
    modulation = modulation.upper()
    
    # 优先使用固定路径
    if modulation in raw_data_paths:
        fixed_path = Path(raw_data_paths[modulation])
        if fixed_path.exists():
            return [fixed_path]
        else:
            print(f"警告: 固定路径不存在: {fixed_path}")
    
    # 如果固定路径不存在，尝试在基础目录中查找
    base_dir = Path(raw_data_base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"数据基础目录不存在: {base_dir}")
    
    # 尝试在调制方式子目录中查找
    mod_patterns = [
        f"{modulation}_*",
        f"*{modulation}*",
        modulation.lower(),
    ]
    
    dat_files = []
    for pattern in mod_patterns:
        # 在基础目录下查找
        found = list(base_dir.glob(f"{pattern}/*.dat"))
        if found:
            dat_files.extend(found)
        # 也在子目录中查找
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                found = list(subdir.glob(f"{pattern}/*.dat"))
                if found:
                    dat_files.extend(found)
    
    if not dat_files:
        raise FileNotFoundError(
            f"未找到 {modulation} 的.dat文件。"
            f"请检查 YAML 配置文件中的 data_generation.raw_data 部分。"
        )
    
    return sorted(set(dat_files))


def read_iq_data_chunked(file_path, start_sample, num_samples, original_sps):
    """按块读取I、Q数据（优化大文件I/O）"""
    num_shorts = num_samples * 2
    start_byte = start_sample * 2 * 2  # 每个样本4字节（2个short）
    
    mmap_data = np.memmap(file_path, dtype=np.int16, mode='r', 
                          offset=start_byte, shape=(num_shorts,))
    
    data_reshaped = mmap_data.reshape(-1, 2)
    complex_data = data_reshaped[:, 0].astype(np.float32) + 1j * data_reshaped[:, 1].astype(np.float32)
    
    return complex_data


def process_slice_for_labeling(signal_slice, modulation, current_sps, rrc_filter_coeff, threshold=6.0):
    """处理单个切片，评估聚类质量，判断是否可解调"""
    target_score = threshold
    
    optimal_bw, best_score_search, phase_history_search, signal_mf, best_offset_search = find_optimal_loop_bandwidth(
        signal_slice, sps=current_sps, fs=FS, modulation=modulation,
        target_score=target_score, min_bw=1e-14, max_bw=1e-5, num_trials=50, verbose=False
    )
    
    slice_best_offset = best_offset_search
    optimal_bw_used = optimal_bw
    
    slice_best_symbols = signal_mf[slice_best_offset::current_sps]
    slice_best_score = evaluate_clustering_quality(slice_best_symbols, modulation)
    
    is_demodulable = slice_best_score >= threshold
    
    return is_demodulable, slice_best_score, optimal_bw_used, slice_best_offset


def process_dat_file(file_path, modulation, max_slices=None, max_samples=None, threshold=6.0, evaluate_demodulable=False):
    """处理单个.dat文件，切片并评估每个切片的可解调性"""
    print(f"\n处理文件: {file_path}")
    
    # 确定sps
    if modulation.upper() == 'QPSK':
        original_sps = 8
    elif modulation.upper() in ['8PSK', '16QAM']:
        original_sps = 16
    else:
        print(f"警告: 未知调制方式 {modulation}，默认使用sps=8")
        original_sps = 8
    
    print(f"调制方式: {modulation}, 原始sps: {original_sps}")
    
    # 获取文件大小，计算总样本数
    file_size = os.path.getsize(file_path)
    total_samples_original = file_size // (2 * 2)  # 每个复数样本占4字节
    
    # 计算下采样后的总样本数
    if original_sps != TARGET_SPS:
        downsample_factor = original_sps // TARGET_SPS
        total_samples_downsampled = total_samples_original // downsample_factor
    else:
        total_samples_downsampled = total_samples_original
    
    # 计算可用的切片数
    num_slices_available = total_samples_downsampled // SLICE_LENGTH
    
    # 确定要处理的切片数
    if max_slices is not None:
        num_slices = min(num_slices_available, max_slices)
    elif max_samples is not None:
        num_slices = min(num_slices_available, max_samples)
    else:
        num_slices = num_slices_available
    
    print(f"文件总大小: {file_size / (1024**3):.2f} GB")
    print(f"总样本数（原始sps）: {total_samples_original}")
    print(f"总样本数（下采样后）: {total_samples_downsampled}")
    print(f"可以切出 {num_slices_available} 片，将处理前 {num_slices} 片")
    
    # 创建RRC滤波器
    current_sps = TARGET_SPS
    rrc_filter_coeff = rrc_filter(0.33, current_sps, 64)
    
    slices_data = []
    
    # 按块处理切片
    chunk_size = 10
    processed_count = 0
    
    for chunk_start in range(0, num_slices, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_slices)
        chunk_slices = chunk_end - chunk_start
        
        # 计算需要读取的原始样本范围
        if original_sps != TARGET_SPS:
            downsample_factor = original_sps // TARGET_SPS
            samples_per_slice_original = SLICE_LENGTH * downsample_factor
        else:
            downsample_factor = 1
            samples_per_slice_original = SLICE_LENGTH
        
        start_sample_original = chunk_start * samples_per_slice_original
        buffer_samples = 64 * downsample_factor * 2
        num_samples_to_read = chunk_slices * samples_per_slice_original + buffer_samples
        num_samples_to_read = min(num_samples_to_read, total_samples_original - start_sample_original)
        
        # 读取当前块
        signal_chunk_original = read_iq_data_chunked(
            file_path, start_sample_original, num_samples_to_read, original_sps
        )
        
        # 下采样
        if original_sps != TARGET_SPS:
            signal_chunk = downsample_to_sps8(signal_chunk_original, original_sps)
        else:
            signal_chunk = signal_chunk_original
        
        # 处理当前块中的每个切片
        for local_idx in range(chunk_slices):
            slice_idx = chunk_start + local_idx
            slice_start_in_chunk = local_idx * SLICE_LENGTH
            slice_end_in_chunk = slice_start_in_chunk + SLICE_LENGTH
            
            if slice_end_in_chunk > len(signal_chunk):
                break
            
            signal_slice_raw = signal_chunk[slice_start_in_chunk:slice_end_in_chunk].copy()
            
            if evaluate_demodulable:
                is_demodulable, clustering_score, optimal_bw, best_offset = process_slice_for_labeling(
                    signal_slice_raw, modulation, current_sps, rrc_filter_coeff, threshold
                )
                
                slices_data.append({
                    'slice_data': signal_slice_raw,
                    'clustering_score': float(clustering_score),
                    'is_demodulable': bool(is_demodulable),
                    'optimal_bw': float(optimal_bw),
                    'best_offset': int(best_offset),
                    'file_path': str(file_path),
                    'file_stem': Path(file_path).stem,
                    'slice_idx': slice_idx
                })
            else:
                slices_data.append({
                    'slice_data': signal_slice_raw,
                    'clustering_score': None,
                    'is_demodulable': None,
                    'optimal_bw': None,
                    'best_offset': None,
                    'file_path': str(file_path),
                    'file_stem': Path(file_path).stem,
                    'slice_idx': slice_idx
                })
            
            processed_count += 1
            
            if max_samples is not None and processed_count >= max_samples:
                print(f"  已达到最大样本数 {max_samples}，停止处理")
                break
        
        if (chunk_start + chunk_size) % 50 == 0 or chunk_end >= num_slices:
            print(f"  已处理 {processed_count}/{num_slices} 个切片")
        
        del signal_chunk_original, signal_chunk
        
        if max_samples is not None and processed_count >= max_samples:
            break
    
    file_metadata = {
        'file_path': str(file_path),
        'file_stem': Path(file_path).stem,
        'modulation': modulation,
        'original_sps': original_sps,
        'target_sps': TARGET_SPS,
        'num_slices': processed_count,
        'slice_length': SLICE_LENGTH
    }
    
    if evaluate_demodulable:
        demodulable_count = sum(1 for s in slices_data if s['is_demodulable'])
        print(f"完成处理: {processed_count} 个切片，可解调: {demodulable_count} 个")
    else:
        print(f"完成处理: {processed_count} 个切片（未评估可解调性）")
    
    return slices_data, file_metadata


def split_data(slices_data, train_ratio=0.9, seed=42):
    """划分数据：随机划分训练集和测试集"""
    np.random.seed(seed)
    n_total = len(slices_data)
    
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    n_train = int(n_total * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return train_indices, test_indices


def _evaluate_single_slice_wrapper(args):
    """包装函数，用于多进程并行处理单个切片"""
    slice_info, modulation, threshold, slice_idx = args
    
    try:
        signal_slice_raw = slice_info['slice_data']
        if not isinstance(signal_slice_raw, np.ndarray):
            signal_slice_raw = np.asarray(signal_slice_raw, dtype=np.complex128)
        
        current_sps = TARGET_SPS
        rrc_filter_coeff = rrc_filter(0.33, current_sps, 64)
        
        is_demodulable, clustering_score, optimal_bw, best_offset = process_slice_for_labeling(
            signal_slice_raw, modulation, current_sps, rrc_filter_coeff, threshold
        )
        
        evaluated_slice = slice_info.copy()
        evaluated_slice['clustering_score'] = float(clustering_score)
        evaluated_slice['is_demodulable'] = bool(is_demodulable)
        evaluated_slice['optimal_bw'] = float(optimal_bw)
        evaluated_slice['best_offset'] = int(best_offset)
        
        return (slice_idx, evaluated_slice)
    except Exception as e:
        print(f"    警告: 切片 {slice_idx} 评估失败: {e}")
        evaluated_slice = slice_info.copy()
        evaluated_slice['clustering_score'] = -np.inf
        evaluated_slice['is_demodulable'] = False
        evaluated_slice['optimal_bw'] = None
        evaluated_slice['best_offset'] = None
        return (slice_idx, evaluated_slice)


def evaluate_test_slices(test_slices, modulation, threshold=6.0, num_workers=None):
    """对测试集的切片进行评估，判断是否可解调"""
    print(f"  评估切片的可解调性...")
    
    if num_workers is None:
        num_workers = min(cpu_count(), 64)
    num_workers = min(num_workers, len(test_slices), 64)
    print(f"  使用 {num_workers} 个进程并行处理 {len(test_slices)} 个切片...")
    
    args_list = [
        (slice_info, modulation, threshold, i)
        for i, slice_info in enumerate(test_slices)
    ]
    
    start_time = time.time()
    evaluated_slices = [None] * len(test_slices)
    
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_evaluate_single_slice_wrapper, args_list)
        
        completed = 0
        for slice_idx, evaluated_slice in results:
            evaluated_slices[slice_idx] = evaluated_slice
            completed += 1
            
            if completed % max(10, len(test_slices) // 20) == 0 or completed == len(test_slices):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(test_slices) - completed) / rate if rate > 0 else 0
                print(f"    已评估 {completed}/{len(test_slices)} 个切片 "
                      f"(速度: {rate:.2f} 切片/秒, 剩余时间: {remaining/60:.1f} 分钟)")
    
    for i, s in enumerate(evaluated_slices):
        if s is None:
            evaluated_slices[i] = {
                'slice_data': test_slices[i].get('slice_data', None),
                'file_stem': test_slices[i].get('file_stem', ''),
                'slice_idx': test_slices[i].get('slice_idx', i),
                'clustering_score': -np.inf,
                'is_demodulable': False,
                'optimal_bw': None,
                'best_offset': None
            }
    
    test_demodulable_indices = np.where([s['is_demodulable'] for s in evaluated_slices])[0]
    test_undemodulable_indices = np.where([not s['is_demodulable'] for s in evaluated_slices])[0]
    
    total_time = time.time() - start_time
    print(f"  切片评估完成: {len(evaluated_slices)} 个切片，可解调: {len(test_demodulable_indices)} 个，"
          f"不可解调: {len(test_undemodulable_indices)} 个")
    print(f"  总耗时: {total_time/60:.2f} 分钟，平均速度: {len(test_slices)/total_time:.2f} 切片/秒")
    
    return evaluated_slices, test_demodulable_indices, test_undemodulable_indices


def save_split_data(slices_data, train_indices, test_indices, train_demodulable_indices, test_demodulable_indices,
                   output_dir, modulation, threshold):
    """保存划分后的元数据"""
    output_dir = Path(output_dir)
    mod_output_dir = output_dir / modulation.lower()
    mod_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_demodulable_count = len(train_demodulable_indices) if train_demodulable_indices is not None else None
    train_non_demodulable_count = (len(train_indices) - train_demodulable_count) if train_demodulable_count is not None else None
    
    split_info = {
        'modulation': modulation,
        'threshold': float(threshold),
        'train_ratio': float(len(train_indices) / len(slices_data)),
        'total_slices': int(len(slices_data)),
        'train_slices': int(len(train_indices)),
        'test_slices': int(len(test_indices)),
        'test_demodulable_slices': int(len(test_demodulable_indices)),
        'train_demodulable_count': int(train_demodulable_count) if train_demodulable_count is not None else None,
        'train_non_demodulable_count': int(train_non_demodulable_count) if train_non_demodulable_count is not None else None,
        'test_demodulable_count': int(len(test_demodulable_indices)),
        'test_non_demodulable_count': int(len(test_indices) - len(test_demodulable_indices)),
        'train_indices': [int(idx) for idx in train_indices.tolist()],
        'test_indices': [int(idx) for idx in test_indices.tolist()],
        'train_demodulable_indices': [int(idx) for idx in train_demodulable_indices.tolist()] if train_demodulable_indices is not None else None,
        'test_demodulable_indices': [int(idx) for idx in test_demodulable_indices.tolist()],
        'train_slice_info': [
            {
                'file_stem': slices_data[idx]['file_stem'],
                'slice_idx': slices_data[idx]['slice_idx'],
                'clustering_score': float(slices_data[idx]['clustering_score']) if slices_data[idx].get('clustering_score') is not None else None,
                'is_demodulable': bool(slices_data[idx]['is_demodulable']) if slices_data[idx].get('is_demodulable') is not None else None,
                'optimal_bw': float(slices_data[idx]['optimal_bw']) if slices_data[idx].get('optimal_bw') is not None else None,
                'best_offset': int(slices_data[idx]['best_offset']) if slices_data[idx].get('best_offset') is not None else None
            }
            for idx in train_indices
        ],
        'test_slice_info': [
            {
                'file_stem': slices_data[idx]['file_stem'],
                'slice_idx': slices_data[idx]['slice_idx'],
                'clustering_score': float(slices_data[idx]['clustering_score']) if slices_data[idx]['clustering_score'] is not None else None,
                'is_demodulable': bool(slices_data[idx]['is_demodulable']) if slices_data[idx]['is_demodulable'] is not None else None,
                'optimal_bw': float(slices_data[idx]['optimal_bw']) if slices_data[idx]['optimal_bw'] is not None else None,
                'best_offset': int(slices_data[idx]['best_offset']) if slices_data[idx]['best_offset'] is not None else None
            }
            for idx in test_indices
        ]
    }
    
    info_file = mod_output_dir / "split_info.json"
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"保存划分信息: {info_file}")
    
    print(f"\n{modulation} 划分统计:")
    print(f"  总切片数: {len(slices_data)}")
    print(f"  训练集: {len(train_indices)} 个切片")
    if train_demodulable_count is not None:
        print(f"    - 可解调: {train_demodulable_count} 个")
        print(f"    - 不可解调: {train_non_demodulable_count} 个")
    print(f"  测试集: {len(test_indices)} 个切片")
    print(f"    - 可解调: {split_info['test_demodulable_count']} 个")
    print(f"    - 不可解调: {split_info['test_non_demodulable_count']} 个")


def save_train_only_data(slices_data, train_indices, train_demodulable_indices, output_dir, modulation):
    """保存仅训练集模式的元数据"""
    output_dir = Path(output_dir)
    mod_output_dir = output_dir / modulation.lower()
    mod_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_demodulable_count = len(train_demodulable_indices) if train_demodulable_indices is not None else None
    train_non_demodulable_count = (len(train_indices) - train_demodulable_count) if train_demodulable_count is not None else None
    
    split_info = {
        'modulation': modulation,
        'mode': 'train_only',
        'total_slices': int(len(slices_data)),
        'train_slices': int(len(train_indices)),
        'train_ratio': 1.0,
        'train_demodulable_count': int(train_demodulable_count) if train_demodulable_count is not None else None,
        'train_non_demodulable_count': int(train_non_demodulable_count) if train_non_demodulable_count is not None else None,
        'train_indices': [int(idx) for idx in train_indices.tolist()],
        'train_demodulable_indices': [int(idx) for idx in train_demodulable_indices.tolist()] if train_demodulable_indices is not None else None,
        'train_slice_info': [
            {
                'file_stem': slices_data[idx]['file_stem'],
                'slice_idx': slices_data[idx]['slice_idx'],
                'clustering_score': float(slices_data[idx]['clustering_score']) if slices_data[idx].get('clustering_score') is not None else None,
                'is_demodulable': bool(slices_data[idx]['is_demodulable']) if slices_data[idx].get('is_demodulable') is not None else None,
                'optimal_bw': float(slices_data[idx]['optimal_bw']) if slices_data[idx].get('optimal_bw') is not None else None,
                'best_offset': int(slices_data[idx]['best_offset']) if slices_data[idx].get('best_offset') is not None else None
            }
            for idx in train_indices
        ]
    }
    
    info_file = mod_output_dir / "split_info.json"
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"保存划分信息: {info_file}")
    
    print(f"\n{modulation} 划分统计（仅训练集模式）:")
    print(f"  总切片数: {len(slices_data)}")
    print(f"  训练集: {len(train_indices)} 个切片")
    if train_demodulable_count is not None:
        print(f"    - 可解调: {train_demodulable_count} 个")
        print(f"    - 不可解调: {train_non_demodulable_count} 个")


# ============= 主函数 =============
def main():
    parser = argparse.ArgumentParser(description='从原始.dat文件切片并分训练集测试集')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='配置文件路径（默认: configs/base_config.yaml）')
    args = parser.parse_args()
    
    # 从 YAML 加载配置
    config = load_config_from_yaml(args.config)
    
    print(f"配置文件: {args.config}")
    print(f"输出目录: {config['output_dir']}")
    print(f"聚类分数阈值: {config['threshold']}")
    print(f"训练集比例: {config['train_ratio']}")
    print(f"随机种子: {config['random_seed']}")
    print(f"模式: {config['mode']}")
    print(f"{'='*60}\n")
    
    # 处理每种调制方式
    for modulation in config['modulation_list']:
        print(f"\n{'='*60}")
        print(f"处理调制方式: {modulation}")
        print(f"{'='*60}")
        
        # 查找.dat文件
        try:
            dat_files = find_dat_file(
                modulation, 
                config['raw_data_base_dir'], 
                config['raw_data_paths']
            )
        except FileNotFoundError as e:
            print(f"错误: {e}")
            continue
        
        print(f"找到 {len(dat_files)} 个.dat文件")
        for f in dat_files:
            print(f"  - {f}")
        
        # 处理所有文件并收集切片数据
        all_slices_data = []
        all_file_metadata = []
        
        for dat_file in dat_files:
            try:
                remaining_samples = None
                if config['max_samples'] is not None:
                    remaining_samples = config['max_samples'] - len(all_slices_data)
                    if remaining_samples <= 0:
                        print(f"已达到最大样本数 {config['max_samples']}，停止处理更多文件")
                        break
                
                slices_data, file_metadata = process_dat_file(
                    dat_file, modulation, 
                    max_slices=config['max_slices_per_file'],
                    max_samples=remaining_samples,
                    threshold=config['threshold'],
                    evaluate_demodulable=False
                )
                all_slices_data.extend(slices_data)
                all_file_metadata.append(file_metadata)
                
                if config['max_samples'] is not None and len(all_slices_data) >= config['max_samples']:
                    print(f"已达到最大样本数 {config['max_samples']}，停止处理更多文件")
                    break
            except Exception as e:
                print(f"错误: 处理文件 {dat_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_slices_data) == 0:
            print(f"警告: {modulation} 没有处理到任何切片，跳过")
            continue
        
        if config['max_samples'] is not None and len(all_slices_data) > config['max_samples']:
            print(f"限制总样本数: 从 {len(all_slices_data)} 个切片中选取前 {config['max_samples']} 个")
            all_slices_data = all_slices_data[:config['max_samples']]
        
        print(f"\n总共收集到 {len(all_slices_data)} 个切片（调制方式: {modulation}）")
        
        if config['mode'] == 'evaluate_test_only':
            # 仅评估测试集模式
            print(f"\n仅评估测试集模式")
            
            train_indices, test_indices = split_data(
                all_slices_data, train_ratio=config['train_ratio'], seed=config['random_seed']
            )
            
            print(f"\n划分结果:")
            print(f"  训练集: {len(train_indices)} 个切片（不处理，不保存）")
            print(f"  测试集: {len(test_indices)} 个切片（将进行评估）")
            
            test_slices = [all_slices_data[idx] for idx in test_indices]
            
            evaluated_test_slices, test_demodulable_indices, test_undemodulable_indices = evaluate_test_slices(
                test_slices, modulation, threshold=config['threshold'], num_workers=config['num_workers']
            )
            
            for i, evaluated_slice in enumerate(evaluated_test_slices):
                all_slices_data[test_indices[i]] = evaluated_slice
            
            mod_output_dir = Path(config['output_dir']) / modulation.lower()
            mod_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n保存测试集...")
            test_file = mod_output_dir / "test_slices.npy"
            np.save(test_file, np.array([s['slice_data'] for s in evaluated_test_slices], dtype=object))
            print(f"✓ 测试集已保存: {test_file}")
            
            test_demodulable_slices = [evaluated_test_slices[idx] for idx in test_demodulable_indices]
            test_demodulable_file = mod_output_dir / "test_demodulable_slices.npy"
            np.save(test_demodulable_file, np.array([s['slice_data'] for s in test_demodulable_slices], dtype=object))
            print(f"✓ 测试集（可解调）已保存: {test_demodulable_file} ({len(test_demodulable_slices)} 个切片)")
            
            test_undemodulable_slices = [evaluated_test_slices[idx] for idx in test_undemodulable_indices]
            test_undemodulable_file = mod_output_dir / "test_undemodulable_slices.npy"
            np.save(test_undemodulable_file, np.array([s['slice_data'] for s in test_undemodulable_slices], dtype=object))
            print(f"✓ 测试集（不可解调）已保存: {test_undemodulable_file} ({len(test_undemodulable_slices)} 个切片)")
            
            save_split_data(
                all_slices_data, train_indices, test_indices, None, test_demodulable_indices,
                config['output_dir'], modulation, config['threshold']
            )
            
        elif config['mode'] == 'train_only':
            # 仅处理训练集
            print(f"\n仅处理训练集模式")
            print(f"  所有 {len(all_slices_data)} 个切片将作为训练集")
            
            train_indices = np.arange(len(all_slices_data))
            
            print(f"\n评估训练集（{len(train_indices)} 个切片）...")
            train_slices = [all_slices_data[idx] for idx in train_indices]
            evaluated_train_slices, train_demodulable_indices, train_undemodulable_indices = evaluate_test_slices(
                train_slices, modulation, threshold=config['threshold'], num_workers=config['num_workers']
            )
            
            for i, evaluated_slice in enumerate(evaluated_train_slices):
                all_slices_data[train_indices[i]] = evaluated_slice
            
            print(f"\n保存训练集...")
            train_file = Path(config['output_dir']) / modulation.lower() / "train_slices.npy"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(train_file, np.array([s['slice_data'] for s in evaluated_train_slices], dtype=object))
            print(f"✓ 训练集已保存: {train_file}")
            
            train_demodulable_slices = [evaluated_train_slices[idx] for idx in train_demodulable_indices]
            train_demodulable_file = Path(config['output_dir']) / modulation.lower() / "train_demodulable_slices.npy"
            np.save(train_demodulable_file, np.array([s['slice_data'] for s in train_demodulable_slices], dtype=object))
            print(f"✓ 训练集（可解调）已保存: {train_demodulable_file} ({len(train_demodulable_slices)} 个切片)")
            
            train_undemodulable_slices = [evaluated_train_slices[idx] for idx in train_undemodulable_indices]
            train_undemodulable_file = Path(config['output_dir']) / modulation.lower() / "train_undemodulable_slices.npy"
            np.save(train_undemodulable_file, np.array([s['slice_data'] for s in train_undemodulable_slices], dtype=object))
            print(f"✓ 训练集（不可解调）已保存: {train_undemodulable_file} ({len(train_undemodulable_slices)} 个切片)")
            
            save_train_only_data(
                all_slices_data, train_indices, train_demodulable_indices,
                config['output_dir'], modulation
            )
        else:
            # 正常模式：划分训练集和测试集
            train_indices, test_indices = split_data(
                all_slices_data, train_ratio=config['train_ratio'], seed=config['random_seed']
            )
            
            # 评估训练集
            print(f"\n评估训练集（{len(train_indices)} 个切片）...")
            train_slices = [all_slices_data[idx] for idx in train_indices]
            evaluated_train_slices, train_demodulable_indices, train_undemodulable_indices = evaluate_test_slices(
                train_slices, modulation, threshold=config['threshold'], num_workers=config['num_workers']
            )
            
            for i, evaluated_slice in enumerate(evaluated_train_slices):
                all_slices_data[train_indices[i]] = evaluated_slice
            
            # 保存训练集
            print(f"\n保存训练集...")
            train_file = Path(config['output_dir']) / modulation.lower() / "train_slices.npy"
            train_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(train_file, np.array([s['slice_data'] for s in evaluated_train_slices], dtype=object))
            print(f"✓ 训练集已保存: {train_file}")
            
            train_demodulable_slices = [evaluated_train_slices[idx] for idx in train_demodulable_indices]
            train_demodulable_file = Path(config['output_dir']) / modulation.lower() / "train_demodulable_slices.npy"
            np.save(train_demodulable_file, np.array([s['slice_data'] for s in train_demodulable_slices], dtype=object))
            print(f"✓ 训练集（可解调）已保存: {train_demodulable_file} ({len(train_demodulable_slices)} 个切片)")
            
            train_undemodulable_slices = [evaluated_train_slices[idx] for idx in train_undemodulable_indices]
            train_undemodulable_file = Path(config['output_dir']) / modulation.lower() / "train_undemodulable_slices.npy"
            np.save(train_undemodulable_file, np.array([s['slice_data'] for s in train_undemodulable_slices], dtype=object))
            print(f"✓ 训练集（不可解调）已保存: {train_undemodulable_file} ({len(train_undemodulable_slices)} 个切片)")
            
            # 评估测试集
            print(f"\n评估测试集（{len(test_indices)} 个切片）...")
            test_slices = [all_slices_data[idx] for idx in test_indices]
            evaluated_test_slices, test_demodulable_indices, test_undemodulable_indices = evaluate_test_slices(
                test_slices, modulation, threshold=config['threshold'], num_workers=config['num_workers']
            )
            
            for i, evaluated_slice in enumerate(evaluated_test_slices):
                all_slices_data[test_indices[i]] = evaluated_slice
            
            # 保存测试集
            print(f"\n保存测试集...")
            test_file = Path(config['output_dir']) / modulation.lower() / "test_slices.npy"
            np.save(test_file, np.array([s['slice_data'] for s in evaluated_test_slices], dtype=object))
            print(f"✓ 测试集已保存: {test_file}")
            
            test_demodulable_slices = [evaluated_test_slices[idx] for idx in test_demodulable_indices]
            test_demodulable_file = Path(config['output_dir']) / modulation.lower() / "test_demodulable_slices.npy"
            np.save(test_demodulable_file, np.array([s['slice_data'] for s in test_demodulable_slices], dtype=object))
            print(f"✓ 测试集（可解调）已保存: {test_demodulable_file} ({len(test_demodulable_slices)} 个切片)")
            
            test_undemodulable_slices = [evaluated_test_slices[idx] for idx in test_undemodulable_indices]
            test_undemodulable_file = Path(config['output_dir']) / modulation.lower() / "test_undemodulable_slices.npy"
            np.save(test_undemodulable_file, np.array([s['slice_data'] for s in test_undemodulable_slices], dtype=object))
            print(f"✓ 测试集（不可解调）已保存: {test_undemodulable_file} ({len(test_undemodulable_slices)} 个切片)")
            
            # 保存元数据
            save_split_data(
                all_slices_data, train_indices, test_indices, train_demodulable_indices, test_demodulable_indices,
                config['output_dir'], modulation, config['threshold']
            )
    
    print(f"\n{'='*60}")
    print("所有处理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
