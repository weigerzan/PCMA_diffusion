"""
从训练集和测试集（可解调）切片生成混合信号对：
- 两两加和，随机幅度比（给第二路作用[0.2,0.9]）
- 可选：评估SNR并添加噪声至目标SNR（仅对可解调训练数据）
- 配对策略：
  * 允许切片复用，但尽量均匀使用不同切片
  * 确保每个切片对(i,j)只用一次（不重复相同的两个切片组合）
  * 只使用训练集内部和测试集内部配对，避免数据泄露
- 分别保存：训练集配对保存到 output_dir/train/，测试集配对保存到 output_dir/test/
- 分shard存储，每个shard默认10k组信号对
- 存储格式对齐generate_sim_dataset.py

使用方式：python generate_mixed_from_splits.py [--config configs/base_config.yaml]
配置从 YAML 文件的 data_generation.generate_mixed 部分读取
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import random
import time
import sys
from multiprocessing import Pool, cpu_count
sys.path.append(str(Path(__file__).parent))


# ============= 配置加载函数 =============
def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'data_generation' not in config:
        raise ValueError("YAML 配置文件中缺少 'data_generation' 部分")
    
    raw_data_cfg = config['data_generation'].get('raw_data', {})
    split_cfg = config['data_generation'].get('split', {})
    generate_mixed_cfg = config['data_generation'].get('generate_mixed', {})
    
    # 处理 amp_range（可能是列表，需要转换为元组）
    amp_range = generate_mixed_cfg.get('amp_range', [0.7, 0.7])
    if isinstance(amp_range, list):
        amp_range = tuple(amp_range)
    
    # 处理 amp_list
    amp_list = generate_mixed_cfg.get('amp_list')
    
    # 构建配置字典
    config_dict = {
        'modulation': generate_mixed_cfg.get('modulation', '8PSK'),
        'mode': generate_mixed_cfg.get('mode', 'both'),
        'output_dir': generate_mixed_cfg.get('output_dir', '/nas/datasets/yixin/PCMA/real_data/8psk'),
        'shard_size': generate_mixed_cfg.get('shard_size', 10000),
        'target_pairs': generate_mixed_cfg.get('target_pairs', 100000),
        'test_target_pairs': generate_mixed_cfg.get('test_target_pairs'),
        'amp_range': amp_range,
        'amp_list': amp_list,
        'sps': generate_mixed_cfg.get('sps', 8),
        'random_seed': generate_mixed_cfg.get('random_seed', 42),
        'train_demodulable': generate_mixed_cfg.get('train_demodulable', True),
        'test_demodulable': generate_mixed_cfg.get('test_demodulable', True),
        'undemodulable': generate_mixed_cfg.get('undemodulable', False),
        'add_noise_to_target_snr': generate_mixed_cfg.get('add_noise_to_target_snr', True),
        'target_snr_db': generate_mixed_cfg.get('target_snr_db', 15.0),
        'filter_type': generate_mixed_cfg.get('filter_type', 'RRC'),
        'num_workers': generate_mixed_cfg.get('num_workers'),
        'num_files_per_amp': generate_mixed_cfg.get('num_files_per_amp', 0),
        'samples_per_file': generate_mixed_cfg.get('samples_per_file', 30),
        # 切片文件基础目录（从 split 配置中获取）
        'slices_base_dir': split_cfg.get('output_dir', '/nas/datasets/yixin/PCMA/real_data'),
    }
    
    return config_dict


# ============= 辅助函数 =============
def find_slices_files(modulation, base_dir, mode):
    """
    根据调制方式和模式自动查找切片文件路径。
    
    参数:
        modulation: 调制方式
        base_dir: 基础目录（split_from_raw_data.py的输出目录）
        mode: 模式 ("train", "test", "both")
    
    返回:
        (train_slices_path, test_slices_path)
    """
    base_path = Path(base_dir)
    mod_dir = base_path / modulation.lower()
    
    train_path = None
    test_path = None
    
    if mode in ["train", "both"]:
        # 优先查找可解调的训练集切片
        train_demod_path = mod_dir / "train_demodulable_slices.npy"
        train_path_normal = mod_dir / "train_slices.npy"
        
        if train_demod_path.exists():
            train_path = train_demod_path
        elif train_path_normal.exists():
            train_path = train_path_normal
        else:
            raise FileNotFoundError(
                f"未找到训练集切片文件。请检查以下路径：\n"
                f"  - {train_demod_path}\n"
                f"  - {train_path_normal}"
            )
    
    if mode in ["test", "both"]:
        # 优先查找可解调的测试集切片
        test_demod_path = mod_dir / "test_demodulable_slices.npy"
        test_path_normal = mod_dir / "test_slices.npy"
        
        if test_demod_path.exists():
            test_path = test_demod_path
        elif test_path_normal.exists():
            test_path = test_path_normal
        else:
            raise FileNotFoundError(
                f"未找到测试集切片文件。请检查以下路径：\n"
                f"  - {test_demod_path}\n"
                f"  - {test_path_normal}"
            )
    
    return train_path, test_path



def energy_normalize_dataset(dataset):
    """
    能量归一化数据集（与generate_sim_dataset.py对齐）
    归一化整个数据集的平均能量
    """
    if not dataset:
        return dataset
    
    energies = [np.mean(np.abs(e['mixsignal']) ** 2) for e in dataset]
    mean_e = np.mean(energies) if energies else 1.0
    scale = np.sqrt(mean_e)
    
    for e in dataset:
        e['mixsignal'] = e['mixsignal'] / scale
        e['rfsignal1'] = e['rfsignal1'] / scale
        e['rfsignal2'] = e['rfsignal2'] / scale
    
    return dataset


def generate_pairs_from_slices(train_slices, test_slices, target_pairs=250000, amp_range=(0.2, 0.9), seed=42):
    """
    从训练集和测试集切片生成混合信号对（优化内存使用）
    - 允许切片复用，但尽量均匀使用
    - 确保每个切片对(i,j)只用一次（不重复相同的两个切片组合）
    - 使用按需生成策略，避免一次性生成所有候选配对（节省内存）
    
    参数：
      - train_slices: 训练集切片列表（可以为空列表，表示不生成训练集配对）
      - test_slices: 测试集（可解调）切片列表（可以为空列表，表示不生成测试集配对）
      - target_pairs: 目标生成的配对数量
      - amp_range: 幅度比范围（作用在第二路）
      - seed: 随机种子
    
    返回：
      - pairs: 列表，每个元素是 (source1, idx1, source2, idx2, amp_ratio)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    pairs = []
    used_pairs = set()  # 记录已使用的切片对，避免重复
    
    # 计算总候选配对数量（用于显示，不实际生成）
    train_candidates = len(train_slices) * (len(train_slices) - 1) if len(train_slices) > 0 else 0
    test_candidates = len(test_slices) * (len(test_slices) - 1) if len(test_slices) > 0 else 0
    total_candidates = train_candidates + test_candidates
    
    print(f"  总候选配对: {total_candidates:,} (训练集: {train_candidates:,}, 测试集: {test_candidates:,})")
    print(f"  目标配对: {target_pairs:,}")
    print(f"  使用按需生成策略（节省内存）...")
    
    # 按需生成配对，避免一次性生成所有候选
    pair_count = 0
    max_attempts = min(target_pairs * 2, total_candidates)  # 最多尝试次数
    attempts = 0
    last_print_count = 0
    last_print_time = None
    start_time = time.time()
    
    # 创建索引列表用于随机采样
    train_indices = list(range(len(train_slices))) if len(train_slices) > 0 else []
    test_indices = list(range(len(test_slices))) if len(test_slices) > 0 else []
    
    print(f"  开始生成配对（最多尝试 {max_attempts:,} 次）...")
    
    while pair_count < target_pairs and attempts < max_attempts:
        attempts += 1
        
        # 随机选择是训练集还是测试集配对
        if len(train_indices) > 0 and len(test_indices) > 0:
            use_train = random.random() < (len(train_indices) / (len(train_indices) + len(test_indices)))
        elif len(train_indices) > 0:
            use_train = True
        elif len(test_indices) > 0:
            use_train = False
        else:
            break
        
        if use_train:
            # 训练集内部配对
            idx1 = random.choice(train_indices)
            idx2 = random.choice(train_indices)
            if idx1 == idx2:
                continue
            source1, source2 = 'train', 'train'
        else:
            # 测试集内部配对
            idx1 = random.choice(test_indices)
            idx2 = random.choice(test_indices)
            if idx1 == idx2:
                continue
            source1, source2 = 'test', 'test'
        
        # 检查这个配对是否已使用（避免重复相同的两个切片组合）
        pair_key = (source1, idx1, source2, idx2)
        if pair_key not in used_pairs:
            amp_ratio = np.random.uniform(*amp_range)
            pairs.append((source1, idx1, source2, idx2, amp_ratio))
            used_pairs.add(pair_key)
            pair_count += 1
            
            # 每生成1000个配对或每5秒输出一次进展
            current_time = time.time()
            should_print = (pair_count - last_print_count >= 1000) or \
                          (last_print_time is None or current_time - last_print_time >= 5)
            
            if should_print:
                elapsed = current_time - start_time
                rate = pair_count / elapsed if elapsed > 0 else 0
                success_rate = (pair_count / attempts * 100) if attempts > 0 else 0
                remaining = target_pairs - pair_count
                eta = remaining / rate if rate > 0 else 0
                
                print(f"    已生成 {pair_count:,}/{target_pairs:,} 个配对 "
                      f"(尝试: {attempts:,}, 成功率: {success_rate:.1f}%, "
                      f"速度: {rate:.0f} 对/秒, 预计剩余: {eta:.0f}秒)")
                
                last_print_count = pair_count
                last_print_time = current_time
    
    elapsed_total = time.time() - start_time
    
    print(f"\n  配对生成完成:")
    print(f"    生成配对: {pair_count:,}/{target_pairs:,}")
    print(f"    总尝试次数: {attempts:,}")
    if attempts > 0:
        print(f"    成功率: {pair_count/attempts*100:.2f}%")
    print(f"    总耗时: {elapsed_total:.1f} 秒")
    if elapsed_total > 0:
        print(f"    平均速度: {pair_count/elapsed_total:.0f} 对/秒")
    
    if pair_count < target_pairs:
        print(f"  警告: 只能生成 {pair_count:,} 个不重复配对，少于目标 {target_pairs:,}")
        print(f"  原因: 尝试了 {attempts:,} 次，可能候选配对已用完或随机采样效率较低")
        print(f"  建议: 如果候选配对充足，可以增加 max_attempts 或使用不同的随机策略")
    
    # 统计每个切片的使用次数（严格隔离检查）
    train_usage = {}
    test_usage = {}
    has_train = len(train_slices) > 0
    has_test = len(test_slices) > 0
    
    for source1, idx1, source2, idx2, _ in pairs:
        # 数据泄露检查：如果只有训练集，不应该有测试集配对
        if not has_test and (source1 == 'test' or source2 == 'test'):
            raise RuntimeError(f"数据泄露检测失败: 在仅训练集模式下发现测试集配对！({source1}, {source2})")
        # 数据泄露检查：如果只有测试集，不应该有训练集配对
        if not has_train and (source1 == 'train' or source2 == 'train'):
            raise RuntimeError(f"数据泄露检测失败: 在仅测试集模式下发现训练集配对！({source1}, {source2})")
        
        if source1 == 'train':
            train_usage[idx1] = train_usage.get(idx1, 0) + 1
        else:
            test_usage[idx1] = test_usage.get(idx1, 0) + 1
        if source2 == 'train':
            train_usage[idx2] = train_usage.get(idx2, 0) + 1
        else:
            test_usage[idx2] = test_usage.get(idx2, 0) + 1
    
    if train_usage:
        avg_train_usage = np.mean(list(train_usage.values()))
        max_train_usage = max(train_usage.values())
        print(f"  训练集切片使用统计: 平均 {avg_train_usage:.2f} 次，最多 {max_train_usage} 次")
    if test_usage:
        avg_test_usage = np.mean(list(test_usage.values()))
        max_test_usage = max(test_usage.values())
        print(f"  测试集切片使用统计: 平均 {avg_test_usage:.2f} 次，最多 {max_test_usage} 次")
    
    return pairs


def _create_entry_worker(args_tuple):
    """
    多进程工作函数：创建数据条目
    参数被打包成元组以支持multiprocessing
    """
    (sig1, sig2, amp_ratio, modulation, sps, source1, idx1, source2, idx2,
     add_noise_to_target_snr, target_snr_db, filter_type, seed) = args_tuple
    
    # 确保输入是numpy数组（multiprocessing传递时可能需要）
    sig1 = np.asarray(sig1, dtype=np.complex128)
    sig2 = np.asarray(sig2, dtype=np.complex128)
    
    return create_entry(sig1, sig2, amp_ratio, modulation, sps, source1, idx1, source2, idx2,
                       add_noise_to_target_snr, target_snr_db, filter_type, seed)


def create_entry(sig1, sig2, amp_ratio, modulation, sps=8, source1='train', idx1=0, source2='train', idx2=0,
                 add_noise_to_target_snr=False, target_snr_db=15.0, filter_type="RRC", seed=None):
    """
    创建数据条目（与generate_sim_dataset.py格式对齐）
    
    参数：
      - sig1: 第一路信号
      - sig2: 第二路信号
      - amp_ratio: 幅度比（作用在第二路）
      - modulation: 调制方式
      - sps: 每符号采样数
      - source1, idx1: 第一路信号的来源和索引
      - source2, idx2: 第二路信号的来源和索引
      - add_noise_to_target_snr: 是否对混合信号添加噪声
      - target_snr_db: 目标SNR（假设信号1和信号2无噪，直接对合路信号加噪至目标SNR）
      - filter_type: 滤波器类型（已不使用，保留以兼容旧代码）
      - seed: 随机种子（用于噪声生成）
    
    返回：
      - entry: 数据条目字典
    """
    # 对齐长度
    min_len = min(len(sig1), len(sig2))
    sig1_aligned = sig1[:min_len]
    sig2_aligned = sig2[:min_len]
    
    # 应用幅度比到第二路
    sig2_scaled = sig2_aligned * amp_ratio
    
    # 混合信号
    mixsignal = sig1_aligned + sig2_scaled
    
    # 噪声添加逻辑（简化版）：
    # 1. 假设信号1和信号2都是无噪的
    # 2. 直接对合路信号加噪使其达到目标SNR
    # 3. 不再评估原始信噪比，不再丢弃低SNR数据
    actual_snr_db = 999.0  # 默认无噪声（不加噪时）
    
    if add_noise_to_target_snr:
        # 计算混合信号的信号功率（假设无噪声）
        signal_power = np.mean(np.abs(mixsignal) ** 2)
        
        # 目标SNR转换为线性值
        target_snr_linear = 10 ** (target_snr_db / 10)
        
        # 计算目标噪声功率：SNR = 信号功率 / 噪声功率
        # 因此：噪声功率 = 信号功率 / SNR
        target_noise_power = signal_power / target_snr_linear
        
        # 生成AWGN噪声
        if seed is not None:
            np.random.seed(seed)
        
        # 噪声标准差（复噪声，实部和虚部各占一半功率）
        noise_std = np.sqrt(target_noise_power / 2)
        noise = noise_std * (np.random.randn(len(mixsignal)) + 1j * np.random.randn(len(mixsignal)))
        
        # 添加噪声到混合信号
        mixsignal = mixsignal + noise
        actual_snr_db = target_snr_db
    
    # 创建条目（与generate_sim_dataset.py格式对齐）
    # params格式: (snr_db, amplitude_ratio, sps, f_off1_str, f_off2_str, phi1_str, phi2_str, delay1_str, delay2_str, mod1_str, mod2_str, ...)
    entry = {
        'mixsignal': mixsignal,
        'rfsignal1': sig1_aligned,
        'rfsignal2': sig2_scaled,  # 注意：这里保存的是缩放后的第二路信号
        'params': (
            float(actual_snr_db),  # snr_db (实际SNR)
            float(amp_ratio),  # amplitude_ratio
            int(sps),  # sps
            'f_off1=0.00Hz',  # f_off1 (无频偏，设为0)
            'f_off2=0.00Hz',  # f_off2 (无频偏，设为0)
            'phi1=0.0000rad',  # phi1 (无相偏，设为0)
            'phi2=0.0000rad',  # phi2 (无相偏，设为0)
            'delay1_samp=0',  # delay1_samp (无时延差，设为0)
            'delay2_samp=0',  # delay2_samp (无时延差，设为0)
            f'mod1={modulation}',  # mod1
            f'mod2={modulation}',  # mod2
            f'source1={source1}_idx{idx1}',  # 第一路来源
            f'source2={source2}_idx{idx2}',  # 第二路来源
        ),
        'bits1': np.array([], dtype=np.int8),  # 实采数据没有比特信息
        'bits2': np.array([], dtype=np.int8),
        'origin_len': 1
    }
    
    # 不再保存SNR统计信息（因为不再评估原始SNR）
    
    return entry


def process_test_pairs_with_amp(test_pairs, test_slices, config, amp_ratio):
    """
    处理测试集配对（使用指定的固定幅度比）
    支持生成多个文件（每个文件30个样本），同时生成无噪和加噪两种版本
    返回保存的文件路径列表
    """
    test_saved_paths = []
    
    # 检查是否是test1_2或test2_2模式（需要生成多个文件，每个文件30个样本）
    is_test12_or_test22 = config.get('num_files_per_amp', 0) > 0
    if is_test12_or_test22:
        num_files = config['num_files_per_amp']
        samples_per_file = config.get('samples_per_file', 30)
        print(f"\n处理测试集配对（幅度比={amp_ratio}，每个文件 {samples_per_file} 组，生成 {num_files} 个文件）...")
    else:
        num_files = 1
        samples_per_file = config['shard_size']
        print(f"\n处理测试集配对（幅度比={amp_ratio}，每个shard {config['shard_size']} 组）...")
    
    # 判断是否需要添加噪声（仅对可解调测试数据，且明确指定了add_noise_to_target_snr）
    add_noise_test = config.get('add_noise_to_target_snr', False) and config.get('test_demodulable', False)
    
    # 创建输出目录
    nonoise_output_dir = Path(config['output_dir']) / "nonoise"
    noise_output_dir = Path(config['output_dir']) / "test" / "demodulable" if config.get('test_demodulable', False) else Path(config['output_dir']) / "test"
    
    if is_test12_or_test22:
        # test1_2 或 test2_2 模式：生成多个文件，同时生成无噪和加噪版本
        print(f"  将同时生成无噪版本（保存到 nonoise/）和加噪版本（保存到 test/demodulable/）")
        if add_noise_test:
            print(f"  加噪版本：将评估两路源信号和混合信号SNR，如果混合信号SNR>{config['target_snr_db']}dB，则添加噪声至{config['target_snr_db']}dB")
            print(f"  如果混合信号SNR<{config['target_snr_db']}dB，则丢弃该配对")
            print(f"  滤波器类型：{config.get('filter_type', 'RRC')}")
        nonoise_output_dir.mkdir(parents=True, exist_ok=True)
        if add_noise_test:
            noise_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个文件生成不同的配对（使用不同的随机种子）
        for file_idx in range(num_files):
            file_seed = config['random_seed'] + file_idx * 1000 + int(amp_ratio * 1000)  # 基于文件索引和幅度比的种子
            
            print(f"\n{'='*60}")
            print(f"生成文件 {file_idx + 1}/{num_files} (幅度比={amp_ratio}, seed={file_seed})")
            print(f"{'='*60}")
            
            # 为当前文件生成配对（每个文件需要 samples_per_file 个配对，但加噪版本可能会丢弃一些，所以生成更多）
            target_pairs_per_file = samples_per_file * 2 if add_noise_test else samples_per_file  # 加噪版本可能需要更多配对
            
            file_pairs_list = generate_pairs_from_slices(
                [], test_slices,
                target_pairs=target_pairs_per_file,
                amp_range=(amp_ratio, amp_ratio),
                seed=file_seed
            )
            file_pairs = [(s1, i1, s2, i2, amp) for s1, i1, s2, i2, amp in file_pairs_list 
                         if s1 == 'test' and s2 == 'test']
            
            # 处理无噪版本
            print(f"  处理无噪版本...")
            nonoise_entries = process_file_pairs(file_pairs, test_slices, config, amp_ratio, 
                                                add_noise=False, file_idx=file_idx, file_seed=file_seed)
            
            # 只取前 samples_per_file 个
            if len(nonoise_entries) > samples_per_file:
                nonoise_entries = nonoise_entries[:samples_per_file]
            
            if len(nonoise_entries) > 0:
                save_path = save_shard(nonoise_entries, file_idx, nonoise_output_dir,
                                      config['modulation'], num_files, (amp_ratio, amp_ratio),
                                      seed=file_seed, is_nonoise=True)
                if save_path:
                    test_saved_paths.append(save_path)
                print(f"  ✅ 无噪版本已保存: {save_path} (样本数: {len(nonoise_entries)})")
            
            # 处理加噪版本（如果启用）
            if add_noise_test:
                print(f"  处理加噪版本...")
                noise_entries = process_file_pairs(file_pairs, test_slices, config, amp_ratio,
                                                   add_noise=True, file_idx=file_idx, file_seed=file_seed)
                
                # 只取前 samples_per_file 个
                if len(noise_entries) > samples_per_file:
                    noise_entries = noise_entries[:samples_per_file]
                
                if len(noise_entries) > 0:
                    save_path = save_shard(noise_entries, file_idx, noise_output_dir,
                                          config['modulation'], num_files, (amp_ratio, amp_ratio),
                                          seed=file_seed, is_nonoise=False)
                    if save_path:
                        test_saved_paths.append(save_path)
                    print(f"  ✅ 加噪版本已保存: {save_path} (样本数: {len(noise_entries)})")
    else:
        # 原有逻辑：单个shard文件
        if add_noise_test:
            print(f"  将评估两路源信号和混合信号SNR，如果混合信号SNR>{config['target_snr_db']}dB，则添加噪声至{config['target_snr_db']}dB")
            print(f"  如果混合信号SNR<{config['target_snr_db']}dB，则丢弃该配对")
            print(f"  滤波器类型：{config.get('filter_type', 'RRC')}")
        else:
            print(f"  直接组合数据，不评估SNR，不添加噪声")
        
        print(f"  使用多进程处理（{config.get('num_workers', cpu_count())} 个工作进程）...")
        
        # 准备多进程参数
        worker_args = []
        for pair_idx, (source1, idx1, source2, idx2, _) in enumerate(test_pairs):
            sig1 = test_slices[idx1]
            sig2 = test_slices[idx2]
            noise_seed = config['random_seed'] + pair_idx if add_noise_test else None
            worker_args.append((
                sig1, sig2, amp_ratio, config['modulation'], config['sps'],  # 使用固定的amp_ratio
                source1, idx1, source2, idx2,
                add_noise_test, config['target_snr_db'], config.get('filter_type', 'RRC'), noise_seed
            ))
        
        # 使用多进程处理
        print(f"  开始并行处理 {len(test_pairs)} 个配对...")
        start_time = time.time()
        
        test_entries = []
        discarded_count = 0  # 统计丢弃的数量
        with Pool(processes=config.get('num_workers', cpu_count())) as pool:
            # 使用imap以便显示进度
            results = pool.imap(_create_entry_worker, worker_args)
            
            # 显示进度
            completed = 0
            total = len(worker_args)
            last_print_time = time.time()
            print_interval = 2.0  # 每2秒打印一次进度
            
            for result in results:
                completed += 1
                
                # 过滤掉None值（SNR低于目标值的被丢弃）
                if result is None:
                    discarded_count += 1
                else:
                    test_entries.append(result)
                
                # 定期打印进度
                current_time = time.time()
                if current_time - last_print_time >= print_interval or completed == total:
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    progress_pct = completed / total * 100
                    valid_count = len(test_entries)
                    print(f"    进度: {completed}/{total} ({progress_pct:.1f}%) | "
                          f"有效: {valid_count} | 丢弃: {discarded_count} | "
                          f"已用: {elapsed:.1f}s | 速度: {rate:.1f} 配对/s | "
                          f"预计剩余: {remaining:.1f}s", flush=True)
                    last_print_time = current_time
        
        elapsed_time = time.time() - start_time
        print(f"  并行处理完成，耗时 {elapsed_time:.2f} 秒（平均 {elapsed_time/len(test_pairs)*1000:.2f} ms/配对）")
        if add_noise_test:
            print(f"  有效配对: {len(test_entries)} (全部保留)")
        else:
            print(f"  有效配对: {len(test_entries)} (全部保留，未评估SNR)")
        
        # 确定输出目录
        if not add_noise_test:
            test_output_dir = nonoise_output_dir
        elif config.get('test_demodulable', False):
            test_output_dir = noise_output_dir
        elif config.get('undemodulable', False):
            test_output_dir = Path(config['output_dir']) / "test" / "undemodulable"
        else:
            test_output_dir = Path(config['output_dir']) / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计SNR信息
        snr_stats = []
        noise_added_count = 0
        
        test_shard_entries = []
        test_shard_idx = 0
        
        for entry in test_entries:
            test_shard_entries.append(entry)
            
            # 统计SNR信息（从entry中提取）
            if add_noise_test:
                actual_snr = entry['params'][0]
                snr_stats.append(actual_snr)
                
                if actual_snr <= config['target_snr_db'] + 0.1:  # 允许0.1dB误差
                    noise_added_count += 1
            
            # 如果达到shard大小，保存（使用当前幅度比）
            if len(test_shard_entries) >= config['shard_size']:
                total_shards = (len(test_pairs) + config['shard_size'] - 1) // config['shard_size']
                save_path = save_shard(test_shard_entries, test_shard_idx, test_output_dir, 
                                      config['modulation'], total_shards, (amp_ratio, amp_ratio))
                if save_path:
                    test_saved_paths.append(save_path)
                test_shard_entries = []
                test_shard_idx += 1
        
        # 保存最后一个shard
        if test_shard_entries:
            total_shards = test_shard_idx + 1
            save_path = save_shard(test_shard_entries, test_shard_idx, test_output_dir, 
                                  config['modulation'], total_shards, (amp_ratio, amp_ratio))
            if save_path:
                test_saved_paths.append(save_path)
        
    # 输出SNR统计信息
    if add_noise_test and snr_stats:
        print(f"\n  SNR统计信息（幅度比={amp_ratio}）:")
        print(f"    最终混合信号平均SNR: {np.mean(snr_stats):.2f} dB")
        print(f"    最终混合信号最小SNR: {np.min(snr_stats):.2f} dB")
        print(f"    最终混合信号最大SNR: {np.max(snr_stats):.2f} dB")
        print(f"    已添加噪声样本数: {noise_added_count}/{len(test_pairs) if test_pairs else 0}")
    
    return test_saved_paths


def process_file_pairs(file_pairs, test_slices, config, amp_ratio, add_noise=False, file_idx=0, file_seed=42):
    """
    处理单个文件的配对，返回数据条目列表
    """
    print(f"    使用多进程处理（{config.get('num_workers', cpu_count())} 个工作进程）...")
    
    # 准备多进程参数
    worker_args = []
    for pair_idx, (source1, idx1, source2, idx2, _) in enumerate(file_pairs):
        sig1 = test_slices[idx1]
        sig2 = test_slices[idx2]
        noise_seed = file_seed + pair_idx if add_noise else None
        worker_args.append((
            sig1, sig2, amp_ratio, config['modulation'], config['sps'],
            source1, idx1, source2, idx2,
            add_noise, config['target_snr_db'], config.get('filter_type', 'RRC'), noise_seed
        ))
    
    # 使用多进程处理
    start_time = time.time()
    
    entries = []
    discarded_count = 0
    with Pool(processes=config.get('num_workers', cpu_count())) as pool:
        results = pool.imap(_create_entry_worker, worker_args)
        
        completed = 0
        total = len(worker_args)
        
        for result in results:
            completed += 1
            
            if result is None:
                discarded_count += 1
            else:
                entries.append(result)
            
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                valid_count = len(entries)
                print(f"      进度: {completed}/{total} | 有效: {valid_count} | 丢弃: {discarded_count} | "
                      f"速度: {rate:.1f} 配对/s", flush=True)
    
    elapsed_time = time.time() - start_time
    if add_noise:
        print(f"    处理完成，耗时 {elapsed_time:.2f} 秒，有效: {len(entries)}, 丢弃: {discarded_count}")
    else:
        print(f"    处理完成，耗时 {elapsed_time:.2f} 秒，有效: {len(entries)}")
    
    return entries


def save_shard(entries, shard_idx, output_dir, modulation, total_shards, amp_range, seed=None, is_nonoise=False):
    """
    保存一个shard（与generate_sim_dataset.py格式对齐）
    
    参数:
        seed: 随机种子（可选，用于test1_2/test2_2模式的文件命名）
        is_nonoise: 是否为无噪版本（用于test1_2/test2_2模式的文件命名）
    """
    if not entries:
        return None
    
    # 移除临时的SNR统计信息（不保存到文件）
    for e in entries:
        e.pop('_snr1_db', None)
        e.pop('_snr2_db', None)
        e.pop('_mix_snr_db', None)
    
    # 归一化
    entries_norm = energy_normalize_dataset(entries)
    
    # 转换为numpy数组（确保是complex128）
    for e in entries_norm:
        e['mixsignal'] = np.asarray(e['mixsignal'], dtype=np.complex128)
        e['rfsignal1'] = np.asarray(e['rfsignal1'], dtype=np.complex128)
        e['rfsignal2'] = np.asarray(e['rfsignal2'], dtype=np.complex128)
    
    # 构建文件名（对齐generate_sim_dataset.py格式）
    amp_min, amp_max = amp_range
    if amp_min == amp_max:
        # 固定幅度比，只显示一个值
        amp_str = f"amp{amp_min:.1f}"
    else:
        # 幅度比范围
        amp_str = f"amp{amp_min:.1f}to{amp_max:.1f}"
    
    # 如果是test1_2/test2_2模式，文件名包含种子信息
    if seed is not None:
        noise_suffix = "nonoise" if is_nonoise else f"snr{int(amp_min*10)}dB"  # 简化命名
        base_name = f"real_{modulation.lower()}_mixed_{amp_str}_{noise_suffix}_N{len(entries_norm)}_seed{seed}_c128"
    else:
        base_name = f"real_{modulation.lower()}_mixed_{amp_str}_shard{shard_idx:02d}_of{total_shards:02d}_c128"
    
    save_path = output_dir / f"{base_name}.pth"
    
    # 保存
    torch.save(entries_norm, save_path)
    
    if seed is not None:
        print(f"📦 已保存文件: {save_path} （样本数 {len(entries_norm)}）")
    else:
        print(f"📦 已保存分片 {shard_idx}/{total_shards}: {save_path} （样本数 {len(entries_norm)}）")
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description='从切片生成混合信号对')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='配置文件路径（默认: configs/base_config.yaml）')
    args = parser.parse_args()
    
    # 从 YAML 加载配置
    config = load_config_from_yaml(args.config)
    
    # 设置num_workers默认值
    if config.get('num_workers') is None:
        config['num_workers'] = min(64, cpu_count())
    else:
        config['num_workers'] = min(config['num_workers'], cpu_count())
    print(f"配置文件: {args.config}")
    print(f"使用 {config['num_workers']} 个工作进程（系统核心数: {cpu_count()}）")
    
    # 自动查找切片文件
    train_slices_path, test_slices_path = find_slices_files(
        config['modulation'], config['slices_base_dir'], config['mode']
    )
    
    print(f"\n调制方式: {config['modulation']}")
    print(f"模式: {config['mode']}")
    if train_slices_path:
        print(f"训练集切片文件: {train_slices_path}")
    if test_slices_path:
        print(f"测试集切片文件: {test_slices_path}")
    print(f"输出目录: {config['output_dir']}")
    print(f"{'='*60}\n")
    
    # 严格隔离：只加载对应模式的切片数据
    train_slices = []
    test_slices = []
    
    if config['mode'] in ['train', 'both']:
        if train_slices_path is None:
            raise ValueError(f"生成训练集配对需要训练集切片文件，但未找到。请检查 {config['slices_base_dir']}/{config['modulation'].lower()}/")
        print(f"加载训练集切片: {train_slices_path}")
        train_slices = np.load(train_slices_path, allow_pickle=True)
        train_slices = [np.asarray(s, dtype=np.complex128) for s in train_slices]
        print(f"  训练集切片数: {len(train_slices)}")
        if config.get('train_demodulable', False):
            print(f"  标记为可解调数据（将保存到 train/demodulable/ 目录）")
    else:
        train_slices = []
        print("  训练集切片: 未加载（mode=test，严格隔离）")
    
    if config['mode'] in ['test', 'both']:
        if test_slices_path is None:
            raise ValueError(f"生成测试集配对需要测试集切片文件，但未找到。请检查 {config['slices_base_dir']}/{config['modulation'].lower()}/")
        print(f"加载测试集切片: {test_slices_path}")
        test_slices = np.load(test_slices_path, allow_pickle=True)
        test_slices = [np.asarray(s, dtype=np.complex128) for s in test_slices]
        print(f"  测试集切片数: {len(test_slices)}")
        if config.get('test_demodulable', False):
            print(f"  标记为可解调数据（将保存到 test/demodulable/ 目录）")
        elif config.get('undemodulable', False):
            print(f"  标记为不可解调数据（将保存到 test/undemodulable/ 目录）")
    else:
        test_slices = []
        print("  测试集切片: 未加载（mode=train，严格隔离）")
    
    # 确定测试集目标配对数量
    if config.get('test_target_pairs') is None:
        test_target_pairs = max(1000, config['target_pairs'] // 10)  # 默认是训练集的10%，至少1000
    else:
        test_target_pairs = config['test_target_pairs']
    
    # 处理amp_range/amp_list参数
    amp_list = config.get('amp_list', None)
    amp_range = config.get('amp_range', (0.2, 0.9))
    
    # 如果amp_range是列表，转换为amp_list
    if isinstance(amp_range, list):
        amp_list = amp_range
        amp_range = None
    elif amp_list is not None:
        # 如果指定了amp_list，忽略amp_range
        amp_range = None
    
    # 处理amp_list参数：如果指定了多个固定幅度比，为每个幅度比生成单独的文件
    if amp_list is not None and len(amp_list) > 0:
        # 只处理测试集（test2_2是测试集数据）
        if config['mode'] in ['test', 'both']:
            print(f"\n检测到amp_list参数，将为 {len(amp_list)} 个幅度比生成单独的文件: {amp_list}")
            
            # 为每个幅度比生成文件
            all_test_saved_paths = []
            for amp_idx, amp_ratio in enumerate(amp_list):
                print(f"\n{'='*60}")
                print(f"处理幅度比 {amp_ratio} ({amp_idx+1}/{len(amp_list)})")
                print(f"{'='*60}")
                
                # 生成配对（使用固定幅度比）
                test_pairs_list = generate_pairs_from_slices(
                    [], test_slices,  # 严格隔离：只传入测试集切片
                    target_pairs=test_target_pairs,
                    amp_range=(amp_ratio, amp_ratio), seed=config['random_seed'] + amp_idx
                )
                test_pairs = [(s1, i1, s2, i2, amp) for s1, i1, s2, i2, amp in test_pairs_list 
                              if s1 == 'test' and s2 == 'test']
                print(f"  测试集配对: {len(test_pairs)} 组（已验证：全部为测试集内部配对）")
                
                # 处理测试集配对（使用当前幅度比）
                test_saved_paths = process_test_pairs_with_amp(
                    test_pairs, test_slices, config, amp_ratio
                )
                all_test_saved_paths.extend(test_saved_paths)
            
            # 设置test_saved_paths为所有幅度比的文件
            test_saved_paths = all_test_saved_paths
            test_pairs = []  # 清空，避免后续重复处理
        else:
            raise ValueError("amp_list 参数仅支持 mode='test' 或 mode='both'（且只处理测试集）")
    
    # 生成配对（常规模式，如果没有使用amp_list）
    train_pairs = []
    if not (amp_list is not None and len(amp_list) > 0):
        test_pairs = []
    
    if config['mode'] in ['train', 'both']:
        print(f"\n生成训练集混合信号对...")
        assert len(test_slices) == 0 or config['mode'] == 'both', "训练集模式下不应有测试集切片"
        train_pairs_list = generate_pairs_from_slices(
            train_slices, [],  # 严格隔离：只传入训练集切片
            target_pairs=config['target_pairs'],
            amp_range=tuple(amp_range) if amp_range is not None else (0.2, 0.9), 
            seed=config['random_seed']
        )
        train_pairs = [(s1, i1, s2, i2, amp) for s1, i1, s2, i2, amp in train_pairs_list 
                       if s1 == 'train' and s2 == 'train']
        if len(train_pairs) != len(train_pairs_list):
            raise RuntimeError(f"数据泄露检测: 发现非训练集配对！训练集配对 {len(train_pairs)} != 总配对 {len(train_pairs_list)}")
        print(f"  训练集配对: {len(train_pairs)} 组（已验证：全部为训练集内部配对）")
    
    if config['mode'] in ['test', 'both']:
        # 如果使用 num_files_per_amp 模式（test1_2/test2_2），跳过常规配对生成
        if config.get('num_files_per_amp', 0) > 0:
            print(f"\n使用 test1_2/test2_2 模式：跳过常规配对生成，将在每个文件中单独生成配对")
            test_pairs = []  # 设置为空，process_test_pairs_with_amp 会为每个文件生成新的配对
        else:
            print(f"\n生成测试集混合信号对...")
            assert len(train_slices) == 0 or config['mode'] == 'both', "测试集模式下不应有训练集切片"
            test_pairs_list = generate_pairs_from_slices(
                [], test_slices,  # 严格隔离：只传入测试集切片
                target_pairs=test_target_pairs,
                amp_range=tuple(amp_range) if amp_range is not None else (0.2, 0.9), 
                seed=config['random_seed'] + 1
            )
            test_pairs = [(s1, i1, s2, i2, amp) for s1, i1, s2, i2, amp in test_pairs_list 
                          if s1 == 'test' and s2 == 'test']
            if len(test_pairs) != len(test_pairs_list):
                raise RuntimeError(f"数据泄露检测: 发现非测试集配对！测试集配对 {len(test_pairs)} != 总配对 {len(test_pairs_list)}")
            print(f"  测试集配对: {len(test_pairs)} 组（已验证：全部为测试集内部配对）")
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_dir = None
    test_output_dir = None
    
    if config['mode'] in ['train', 'both']:
        if config.get('train_demodulable', False):
            train_output_dir = output_dir / "train" / "demodulable"
        else:
            train_output_dir = output_dir / "train"
        train_output_dir.mkdir(parents=True, exist_ok=True)
    
    if config['mode'] in ['test', 'both']:
        add_noise_test = config.get('add_noise_to_target_snr', False) and config.get('test_demodulable', False)
        if not add_noise_test:
            test_output_dir = output_dir / "nonoise"
        elif config.get('test_demodulable', False):
            test_output_dir = output_dir / "test" / "demodulable"
        elif config.get('undemodulable', False):
            test_output_dir = output_dir / "test" / "undemodulable"
        else:
            test_output_dir = output_dir / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理训练集配对
    train_saved_paths = []
    if train_pairs:
        print(f"\n处理训练集配对（每个shard {config['shard_size']} 组）...")
        
        # 判断是否需要添加噪声（仅对可解调训练数据）
        add_noise = config.get('add_noise_to_target_snr', False) and config.get('train_demodulable', False)
        if add_noise:
            print(f"  将评估两路源信号和混合信号SNR，如果混合信号SNR>{config['target_snr_db']}dB，则添加噪声至{config['target_snr_db']}dB")
            print(f"  如果混合信号SNR<{config['target_snr_db']}dB，则丢弃该配对")
            print(f"  滤波器类型：{config.get('filter_type', 'RRC')}")
        
        print(f"  使用多进程处理（{config['num_workers']} 个工作进程）...")
        
        # 准备多进程参数
        worker_args = []
        for pair_idx, (source1, idx1, source2, idx2, amp_ratio) in enumerate(train_pairs):
            sig1 = train_slices[idx1]
            sig2 = train_slices[idx2]
            noise_seed = config['random_seed'] + pair_idx if add_noise else None
            worker_args.append((
                sig1, sig2, amp_ratio, config['modulation'], config['sps'],
                source1, idx1, source2, idx2,
                add_noise, config['target_snr_db'], config.get('filter_type', 'RRC'), noise_seed
            ))
        
        # 使用多进程处理
        print(f"  开始并行处理 {len(train_pairs)} 个配对...")
        start_time = time.time()
        
        train_entries = []
        discarded_count = 0  # 统计丢弃的数量
        with Pool(processes=config.get('num_workers', cpu_count())) as pool:
            # 使用imap以便显示进度
            results = pool.imap(_create_entry_worker, worker_args)
            
            # 显示进度
            completed = 0
            total = len(worker_args)
            last_print_time = time.time()
            print_interval = 2.0  # 每2秒打印一次进度
            
            for result in results:
                completed += 1
                
                # 过滤掉None值（SNR低于目标值的被丢弃）
                if result is None:
                    discarded_count += 1
                else:
                    train_entries.append(result)
                
                # 定期打印进度
                current_time = time.time()
                if current_time - last_print_time >= print_interval or completed == total:
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    progress_pct = completed / total * 100
                    valid_count = len(train_entries)
                    print(f"    进度: {completed}/{total} ({progress_pct:.1f}%) | "
                          f"有效: {valid_count} | 丢弃: {discarded_count} | "
                          f"已用: {elapsed:.1f}s | 速度: {rate:.1f} 配对/s | "
                          f"预计剩余: {remaining:.1f}s", flush=True)
                    last_print_time = current_time
        
        elapsed_time = time.time() - start_time
        print(f"  并行处理完成，耗时 {elapsed_time:.2f} 秒（平均 {elapsed_time/len(train_pairs)*1000:.2f} ms/配对）")
        print(f"  有效配对: {len(train_entries)} (全部保留)")
        
        # 统计SNR信息
        snr_stats = []  # 统计混合信号SNR信息
        noise_added_count = 0
        
        train_shard_entries = []
        train_shard_idx = 0
        
        for entry in train_entries:
            train_shard_entries.append(entry)
            
            # 统计SNR信息（从entry中提取）
            if add_noise:
                actual_snr = entry['params'][0]
                snr_stats.append(actual_snr)
                
                if actual_snr <= config['target_snr_db'] + 0.1:  # 允许0.1dB误差
                    noise_added_count += 1
            
            # 如果达到shard大小，保存
            if len(train_shard_entries) >= config['shard_size']:
                total_shards = (len(train_pairs) + config['shard_size'] - 1) // config['shard_size']
                save_path = save_shard(train_shard_entries, train_shard_idx, train_output_dir, 
                                      config['modulation'], total_shards, tuple(amp_range) if amp_range is not None else (0.2, 0.9))
                if save_path:
                    train_saved_paths.append(save_path)
                train_shard_entries = []
                train_shard_idx += 1
        
        # 保存最后一个shard
        if train_shard_entries:
            total_shards = train_shard_idx + 1
            save_path = save_shard(train_shard_entries, train_shard_idx, train_output_dir, 
                                  config['modulation'], total_shards, tuple(amp_range) if amp_range is not None else (0.2, 0.9))
            if save_path:
                train_saved_paths.append(save_path)
        
        # 输出SNR统计信息
        if add_noise and snr_stats:
            print(f"\n  SNR统计信息:")
            print(f"    最终混合信号平均SNR: {np.mean(snr_stats):.2f} dB")
            print(f"    最终混合信号最小SNR: {np.min(snr_stats):.2f} dB")
            print(f"    最终混合信号最大SNR: {np.max(snr_stats):.2f} dB")
            print(f"    已添加噪声样本数: {noise_added_count}/{len(train_pairs)}")

    
    # 处理测试集配对
    test_saved_paths = []
    
    # 如果使用 num_files_per_amp 且没有使用 amp_list（test1_2 模式）
    if config.get('num_files_per_amp', 0) > 0 and (amp_list is None or len(amp_list) == 0):
        # test1_2 模式：单个幅度比，生成多个文件
        if isinstance(amp_range, tuple):
            amp_ratio = amp_range[0] if amp_range[0] == amp_range[1] else amp_range[0]
        else:
            amp_ratio = amp_range[0] if isinstance(amp_range, list) and len(amp_range) > 0 else 0.7
        print(f"\n{'='*60}")
        print(f"处理 test1_2 模式（幅度比={amp_ratio}，每个文件 {config.get('samples_per_file', 30)} 个样本，生成 {config['num_files_per_amp']} 个文件）")
        print(f"{'='*60}")
        
        # 直接调用 process_test_pairs_with_amp，传入空的 test_pairs（函数内部会为每个文件生成配对）
        test_saved_paths = process_test_pairs_with_amp(
            [], test_slices, config, amp_ratio
        )
    
    # 常规模式（如果没有使用amp_list，且没有使用num_files_per_amp）
    elif test_pairs and not (amp_list is not None and len(amp_list) > 0) and config.get('num_files_per_amp', 0) == 0:
        print(f"\n处理测试集配对（每个shard {config['shard_size']} 组）...")
        
        # 判断是否需要添加噪声（仅对可解调测试数据，且明确指定了add_noise_to_target_snr）
        add_noise_test = config.get('add_noise_to_target_snr', False) and config.get('test_demodulable', False)
        if add_noise_test:
            print(f"  将假设信号1和信号2无噪，直接对合路信号加噪至{config['target_snr_db']}dB")
        else:
            print(f"  直接组合数据，不添加噪声")
        
        print(f"  使用多进程处理（{config['num_workers']} 个工作进程）...")
        
        # 准备多进程参数
        worker_args = []
        for pair_idx, (source1, idx1, source2, idx2, amp_ratio) in enumerate(test_pairs):
            sig1 = test_slices[idx1]
            sig2 = test_slices[idx2]
            noise_seed = config['random_seed'] + pair_idx + len(train_pairs) if add_noise_test else None
            worker_args.append((
                sig1, sig2, amp_ratio, config['modulation'], config['sps'],
                source1, idx1, source2, idx2,
                add_noise_test, config['target_snr_db'], config.get('filter_type', 'RRC'), noise_seed
            ))
        
        # 使用多进程处理
        print(f"  开始并行处理 {len(test_pairs)} 个配对...")
        start_time = time.time()
        
        test_entries = []
        discarded_count = 0  # 统计丢弃的数量
        with Pool(processes=config.get('num_workers', cpu_count())) as pool:
            # 使用imap以便显示进度
            results = pool.imap(_create_entry_worker, worker_args)
            
            # 显示进度
            completed = 0
            total = len(worker_args)
            last_print_time = time.time()
            print_interval = 2.0  # 每2秒打印一次进度
            
            for result in results:
                completed += 1
                
                # 过滤掉None值（SNR低于目标值的被丢弃）
                if result is None:
                    discarded_count += 1
                else:
                    test_entries.append(result)
                
                # 定期打印进度
                current_time = time.time()
                if current_time - last_print_time >= print_interval or completed == total:
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    progress_pct = completed / total * 100
                    valid_count = len(test_entries)
                    print(f"    进度: {completed}/{total} ({progress_pct:.1f}%) | "
                          f"有效: {valid_count} | 丢弃: {discarded_count} | "
                          f"已用: {elapsed:.1f}s | 速度: {rate:.1f} 配对/s | "
                          f"预计剩余: {remaining:.1f}s", flush=True)
                    last_print_time = current_time
        
        elapsed_time = time.time() - start_time
        print(f"  并行处理完成，耗时 {elapsed_time:.2f} 秒（平均 {elapsed_time/len(test_pairs)*1000:.2f} ms/配对）")
        if add_noise_test:
            print(f"  有效配对: {len(test_entries)} (全部保留)")
        else:
            print(f"  有效配对: {len(test_entries)} (全部保留，未评估SNR)")
        
        # 统计SNR信息
        snr_stats = []  # 统计混合信号SNR信息
        noise_added_count = 0
        
        test_shard_entries = []
        test_shard_idx = 0
        
        for entry in test_entries:
            test_shard_entries.append(entry)
            
            # 统计SNR信息（从entry中提取）
            if add_noise_test:
                actual_snr = entry['params'][0]
                snr_stats.append(actual_snr)
                
                if actual_snr <= config['target_snr_db'] + 0.1:  # 允许0.1dB误差
                    noise_added_count += 1
            
            # 如果达到shard大小，保存
            if len(test_shard_entries) >= config['shard_size']:
                total_shards = (len(test_pairs) + config['shard_size'] - 1) // config['shard_size']
                save_path = save_shard(test_shard_entries, test_shard_idx, test_output_dir, 
                                      config['modulation'], total_shards, tuple(amp_range) if amp_range is not None else (0.2, 0.9))
                if save_path:
                    test_saved_paths.append(save_path)
                test_shard_entries = []
                test_shard_idx += 1
        
        # 保存最后一个shard
        if test_shard_entries:
            total_shards = test_shard_idx + 1
            save_path = save_shard(test_shard_entries, test_shard_idx, test_output_dir, 
                                  config['modulation'], total_shards, tuple(amp_range) if amp_range is not None else (0.2, 0.9))
            if save_path:
                test_saved_paths.append(save_path)
        
        # 输出SNR统计信息
        if add_noise_test and snr_stats:
            print(f"\n  SNR统计信息:")
            print(f"    最终混合信号平均SNR: {np.mean(snr_stats):.2f} dB")
            print(f"    最终混合信号最小SNR: {np.min(snr_stats):.2f} dB")
            print(f"    最终混合信号最大SNR: {np.max(snr_stats):.2f} dB")
            print(f"    已添加噪声样本数: {noise_added_count}/{len(test_pairs)}")
    
    # 保存元数据
    metadata = {
        'modulation': config['modulation'],
        'mode': config['mode'],
        'train_slices_file': str(train_slices_path) if train_slices_path else None,
        'test_slices_file': str(test_slices_path) if test_slices_path else None,
        'train_pairs': len(train_pairs),
        'test_pairs': len(test_pairs) if 'test_pairs' in locals() else 0,
        'shard_size': config['shard_size'],
        'train_shards': len(train_saved_paths),
        'test_shards': len(test_saved_paths),
        'amp_range': list(amp_range) if amp_range is not None else None,
        'amp_list': amp_list if amp_list is not None else None,
        'sps': config['sps'],
        'seed': config['random_seed'],
        'train_demodulable': config.get('train_demodulable', False),
        'test_demodulable': config.get('test_demodulable', False),
        'test_undemodulable': config.get('undemodulable', False),
        'add_noise_to_target_snr': config.get('add_noise_to_target_snr', False),
        'target_snr_db': config.get('target_snr_db') if config.get('add_noise_to_target_snr', False) else None,
        'filter_type': config.get('filter_type') if config.get('add_noise_to_target_snr', False) else None,
        'generated_at': str(datetime.now()),
        'train_shard_files': [str(p) for p in train_saved_paths],
        'test_shard_files': [str(p) for p in test_saved_paths]
    }
    
    # 在metadata文件名中包含amp_range信息，避免覆盖
    if amp_range is not None:
        amp_min, amp_max = amp_range
        metadata_filename = f"metadata_amp{amp_min:.1f}to{amp_max:.1f}.json"
    elif amp_list is not None:
        amp_str = "_".join([f"{a:.1f}" for a in amp_list])
        metadata_filename = f"metadata_ampList_{amp_str}.json"
    else:
        metadata_filename = "metadata.json"
    metadata_path = output_dir / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n保存元数据: {metadata_path}")
    
    print(f"\n{'='*60}")
    print(f"完成！")
    if train_pairs:
        print(f"  训练集配对: {len(train_pairs)} 组，保存了 {len(train_saved_paths)} 个shard文件")
        print(f"  训练集目录: {train_output_dir}")
        if config.get('train_demodulable', False):
            print(f"  （可解调数据模式）")
    if 'test_pairs' in locals() and test_pairs:
        print(f"  测试集配对: {len(test_pairs)} 组，保存了 {len(test_saved_paths)} 个shard文件")
        print(f"  测试集目录: {test_output_dir}")
        if config.get('test_demodulable', False):
            print(f"  （可解调数据模式）")
        elif config.get('undemodulable', False):
            print(f"  （不可解调数据模式）")
    print(f"输出目录: {output_dir}")
    print(f"模式: {config['mode']}（严格隔离：{'训练集和测试集' if config['mode'] == 'both' else '仅' + ('训练集' if config['mode'] == 'train' else '测试集')}）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

