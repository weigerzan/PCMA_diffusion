#!/usr/bin/env python3
"""
检查GPU设置和可用性
用于调试GPU配置问题
"""

import os
import sys
import torch

def check_gpu():
    print("=" * 60)
    print("GPU 配置检查")
    print("=" * 60)
    
    # 检查环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 检查PyTorch是否能检测到CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"检测到的GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 检查当前设备
        current_device = torch.cuda.current_device()
        print(f"\n当前使用的GPU: {current_device}")
        print(f"当前GPU名称: {torch.cuda.get_device_name(current_device)}")
        
        # 测试GPU计算
        print("\n测试GPU计算...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU计算测试成功")
        except Exception as e:
            print(f"✗ GPU计算测试失败: {e}")
    else:
        print("警告: CUDA不可用，无法使用GPU")
    
    print("=" * 60)
    
    # 解释CUDA_VISIBLE_DEVICES的工作原理
    if cuda_visible != '未设置':
        print("\n说明:")
        print(f"  你设置了 CUDA_VISIBLE_DEVICES={cuda_visible}")
        print(f"  这意味着物理GPU {cuda_visible} 会被映射为逻辑GPU 0")
        print(f"  在nvitop中，应该监控物理GPU {cuda_visible} 的使用情况")
        print(f"  在PyTorch中，使用 torch.cuda.device(0) 会使用物理GPU {cuda_visible}")


if __name__ == '__main__':
    check_gpu()

