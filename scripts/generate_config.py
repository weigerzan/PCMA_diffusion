#!/usr/bin/env python3
"""
从基础配置文件生成新的配置文件
根据调制方式和数据类型更新配置，并设置默认的数据路径
"""

import yaml
import sys
import argparse
from pathlib import Path
import shutil


def generate_config(base_config_path, output_config_path, modulation, data_type):
    """
    从基础配置生成新的配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        output_config_path: 输出配置文件路径
        modulation: 调制方式 (QPSK, 8PSK, 16QAM)
        data_type: 数据类型 (real, sim)
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    modulation = modulation.upper()
    data_type = data_type.lower()
    
    # 更新数据生成配置
    if 'data_generation' in config:
        # 更新仿真数据配置
        if 'generate_sim' in config['data_generation']:
            config['data_generation']['generate_sim']['modulation1'] = modulation
            config['data_generation']['generate_sim']['modulation2'] = modulation
        
        # 更新真实数据配置
        if 'split' in config['data_generation']:
            config['data_generation']['split']['modulation_list'] = [modulation]
        
        if 'generate_mixed' in config['data_generation']:
            config['data_generation']['generate_mixed']['modulation'] = modulation
    
    # 更新数据配置
    if 'data' in config:
        config['data']['modulation'] = modulation
        
        # 根据数据类型设置默认数据路径（占位符，后续会被update_data_paths.py更新）
        if data_type == 'sim':
            # 仿真数据：从generate_sim的save_dir获取
            save_dir = config['data_generation'].get('generate_sim', {}).get('save_dir', '/nas/datasets/yixin/PCMA/sim_data')
            # 默认10个文件，前9个训练，最后1个测试
            config['data']['train'] = {
                'base': save_dir,
                'shard_list': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'pattern': f'{modulation}-{modulation}_*_shard{{idx:02d}}_of10_*.pth'  # 占位符，会被更新
            }
            config['data']['test'] = {
                'paths': []  # 占位符，会被更新为最后一个shard
            }
        elif data_type == 'real':
            # 真实数据：从generate_mixed的output_dir获取
            output_dir = config['data_generation'].get('generate_mixed', {}).get('output_dir', f'/nas/datasets/yixin/PCMA/real_data/{modulation.lower()}')
            # 真实数据有train和test目录
            config['data']['train'] = {
                'base': f'{output_dir}/train',
                'shard_list': [],  # 占位符，会被更新
                'pattern': ''  # 占位符，会被更新
            }
            config['data']['test'] = {
                'paths': []  # 占位符，会被更新
            }
    
    # 更新训练输出目录（添加实验标识）
    if 'training' in config:
        base_output = config['training'].get('output_dir', 'results/DDPM-PCMA')
        experiment_name = Path(output_config_path).stem
        config['training']['output_dir'] = f'{base_output}-{experiment_name}'
    
    # 更新采样输出目录
    if 'sampling' in config:
        base_output = config['sampling'].get('output_dir', '/nas/datasets/yixin/PCMA/diffusion_prediction')
        experiment_name = Path(output_config_path).stem
        config['sampling']['output_dir'] = f'{base_output}/{experiment_name}'
    
    # 确保输出目录存在
    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写回文件
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ 配置文件已生成: {output_config_path}")
    print(f"  调制方式: {modulation}")
    print(f"  数据类型: {data_type}")


def main():
    parser = argparse.ArgumentParser(description='从基础配置生成新的配置文件')
    parser.add_argument('--base-config', type=str, required=True, help='基础配置文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出配置文件路径')
    parser.add_argument('--modulation', type=str, required=True, 
                       choices=['QPSK', '8PSK', '16QAM', 'qpsk', '8psk', '16qam'],
                       help='调制方式')
    parser.add_argument('--data-type', type=str, required=True,
                       choices=['real', 'sim'],
                       help='数据类型 (real/sim)')
    
    args = parser.parse_args()
    
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        print(f"错误: 基础配置文件不存在: {base_config_path}")
        sys.exit(1)
    
    generate_config(args.base_config, args.output, args.modulation, args.data_type)


if __name__ == '__main__':
    main()

