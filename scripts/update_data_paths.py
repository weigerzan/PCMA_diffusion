#!/usr/bin/env python3
"""
更新配置文件中的数据路径
根据实际生成的数据文件自动更新训练和测试数据路径
"""

import yaml
import sys
import argparse
from pathlib import Path
import re


def find_sim_data_files(save_dir, modulation):
    """
    查找仿真数据文件
    
    Args:
        save_dir: 保存目录
        modulation: 调制方式
    
    Returns:
        (train_files, test_files): 训练和测试文件列表
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return [], []
    
    modulation = modulation.upper()
    # 查找匹配的文件：{modulation}-{modulation}_*.pth
    pattern = f"{modulation}-{modulation}_*.pth"
    all_files = sorted(save_dir.glob(pattern))
    
    if not all_files:
        return [], []
    
    # 默认10个文件，前9个训练，最后1个测试
    # 但实际可能不是10个，所以找到所有shard，最后一个作为测试
    train_files = []
    test_files = []
    
    # 按shard编号排序
    shard_files = {}
    for f in all_files:
        match = re.search(r'shard(\d+)_of(\d+)', f.name)
        if match:
            shard_idx = int(match.group(1))
            total_shards = int(match.group(2))
            shard_files[shard_idx] = (f, total_shards)
    
    if shard_files:
        sorted_shards = sorted(shard_files.items())
        total_shards = sorted_shards[0][1][1]  # 从第一个文件获取总数
        
        # 前N-1个作为训练，最后1个作为测试
        for shard_idx, (file_path, _) in sorted_shards[:-1]:
            train_files.append(str(file_path))
        
        if sorted_shards:
            test_files.append(str(sorted_shards[-1][1][0]))
    
    return train_files, test_files


def find_real_data_files(output_dir, modulation):
    """
    查找真实数据文件
    
    Args:
        output_dir: 输出目录
        modulation: 调制方式
    
    Returns:
        (train_files, test_files): 训练和测试文件列表
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return [], []
    
    modulation_lower = modulation.lower()
    train_files = []
    test_files = []
    
    # 查找训练数据：train/ 或 train/demodulable/ 目录
    train_dirs = [
        output_dir / 'train',
        output_dir / 'train' / 'demodulable',
        output_dir / 'train' / 'nonoise',
    ]
    
    for train_dir in train_dirs:
        if train_dir.exists():
            pattern = f"real_{modulation_lower}_mixed_*.pth"
            found = sorted(train_dir.glob(pattern))
            if found:
                train_files = [str(f) for f in found]
                break
    
    # 查找测试数据：test/ 或 test/demodulable/ 目录
    test_dirs = [
        output_dir / 'test',
        output_dir / 'test' / 'demodulable',
        output_dir / 'test' / 'nonoise',
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            pattern = f"real_{modulation_lower}_mixed_*.pth"
            found = sorted(test_dir.glob(pattern))
            if found:
                test_files = [str(f) for f in found]
                break
    
    return train_files, test_files


def update_data_paths(config_path, data_type):
    """
    更新配置文件中的数据路径
    
    Args:
        config_path: 配置文件路径
        data_type: 数据类型 (real, sim)
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_type = data_type.lower()
    modulation = config.get('data', {}).get('modulation', '8PSK')
    
    if 'data' not in config:
        print("错误: 配置文件中缺少 'data' 部分")
        return False
    
    updated = False
    
    if data_type == 'sim':
        # 仿真数据
        save_dir = config.get('data_generation', {}).get('generate_sim', {}).get('save_dir', '')
        if save_dir:
            train_files, test_files = find_sim_data_files(save_dir, modulation)
            
            if train_files:
                # 从第一个文件提取pattern
                sample_file = Path(train_files[0])
                match = re.search(r'shard(\d+)_of(\d+)', sample_file.name)
                
                if match:
                    total_shards = int(match.group(2))
                    # 构建pattern
                    pattern = sample_file.name
                    # 替换shard编号为占位符
                    pattern = re.sub(r'shard\d+_of\d+', f'shard{{idx:02d}}_of{total_shards:02d}', pattern)
                    
                    # 计算shard_list
                    shard_list = []
                    for f in train_files:
                        match = re.search(r'shard(\d+)_of', Path(f).name)
                        if match:
                            shard_list.append(int(match.group(1)))
                    shard_list = sorted(shard_list)
                    
                    config['data']['train'] = {
                        'base': str(sample_file.parent),
                        'shard_list': shard_list,
                        'pattern': pattern
                    }
                    print(f"  ✓ 已更新训练数据路径: {sample_file.parent} ({len(train_files)} 个文件)")
                    updated = True
                else:
                    print(f"  ⚠ 无法解析文件名格式: {sample_file.name}")
            
            if test_files:
                config['data']['test'] = {
                    'paths': test_files
                }
                print(f"  ✓ 已更新测试数据路径: {len(test_files)} 个文件")
                updated = True
            else:
                print(f"  ⚠ 未找到测试数据文件")
        else:
            print(f"  ⚠ 配置文件中未找到 generate_sim.save_dir")
    
    elif data_type == 'real':
        # 真实数据
        output_dir = config.get('data_generation', {}).get('generate_mixed', {}).get('output_dir', '')
        if output_dir:
            train_files, test_files = find_real_data_files(output_dir, modulation)
            
            if train_files:
                # 从第一个文件提取pattern
                sample_file = Path(train_files[0])
                match = re.search(r'shard(\d+)_of(\d+)', sample_file.name)
                
                if match:
                    total_shards = int(match.group(2))
                    # 构建pattern
                    pattern = sample_file.name
                    pattern = re.sub(r'shard\d+_of\d+', f'shard{{idx:02d}}_of{total_shards:02d}', pattern)
                    
                    # 计算shard_list
                    shard_list = []
                    for f in train_files:
                        match = re.search(r'shard(\d+)_of', Path(f).name)
                        if match:
                            shard_list.append(int(match.group(1)))
                    shard_list = sorted(shard_list)
                    
                    config['data']['train'] = {
                        'base': str(sample_file.parent),
                        'shard_list': shard_list,
                        'pattern': pattern
                    }
                    print(f"  ✓ 已更新训练数据路径: {sample_file.parent} ({len(train_files)} 个文件)")
                    updated = True
                else:
                    # 如果无法解析，使用paths方式
                    config['data']['train'] = {
                        'paths': train_files
                    }
                    print(f"  ✓ 已更新训练数据路径: {len(train_files)} 个文件（使用paths方式）")
                    updated = True
            
            if test_files:
                config['data']['test'] = {
                    'paths': test_files
                }
                print(f"  ✓ 已更新测试数据路径: {len(test_files)} 个文件")
                updated = True
            else:
                print(f"  ⚠ 未找到测试数据文件")
        else:
            print(f"  ⚠ 配置文件中未找到 generate_mixed.output_dir")
    
    if updated:
        # 写回文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✓ 配置文件已更新: {config_path}")
        return True
    else:
        print(f"⚠ 未找到数据文件，请确保数据已生成")
        return False


def main():
    parser = argparse.ArgumentParser(description='更新配置文件中的数据路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data-type', type=str, required=True,
                       choices=['real', 'sim'],
                       help='数据类型 (real/sim)')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    success = update_data_paths(args.config, args.data_type)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

