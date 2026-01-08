# Diffusion Models for PCMA Separation
**Developer:** Jiawei Zhang  
**Date:** 2025/12/30  

## Overview
本仓库为 **PCMA Diffusion** 的官方实现，使用扩散模型完成 PCMA 信号的分离与重建。  
主要依赖为 **PyTorch** 与 **Diffusers==0.35.2**，其余依赖可根据报错按需安装。

---

## Configuration
项目使用 YAML 配置文件管理参数。  
参考示例：`configs/base_config.yaml`  
配置文件包含以下部分：
- `data_generation`: 数据生成配置（原始数据处理和混合信号对生成）
- `model`: 模型配置
- `data`: 训练/测试数据路径配置
- `training`: 训练超参数配置
- `sampling`: 推理采样配置

---

## 完整使用流程

### Step 0. 数据生成

项目支持两种数据生成方式：
1. **仿真数据生成**：使用 `generate_sim_dataset.py` 生成仿真混合信号对
2. **真实数据生成**：从采集的原始数据生成训练数据集（两个步骤）

---

### Step 0A. 仿真数据生成

```bash
python generate_data/generate_sim_dataset.py --config configs/base_config.yaml
```

**功能说明：**
- 生成仿真混合信号对（两路信号加和）
- 支持多种调制方式、SNR、幅度比、频偏、相位、时延等参数配置
- 参数可以设置为固定值、范围（均匀分布）或列表（随机选择）
- 分shard保存，便于管理大量数据

**配置示例（在 `configs/base_config.yaml` 中）：**
```yaml
data_generation:
  generate_sim:
    num_samples: 100000  # 总样本数
    shard_size: 10000  # 每个shard的样本数（0表示不分片）
    save_dir: /nas/datasets/yixin/PCMA/sim_data  # 保存目录
    save_complex64: true  # 是否保存为complex64（节省空间）
    random_seed: 42  # 随机种子
    
    # 调制方式（可以是字符串或列表）
    modulation1: 8PSK  # 第一路调制，或 ["QPSK", "8PSK"] 表示随机选择
    modulation2: 8PSK  # 第二路调制
    
    # 信道参数
    # 参数设置方式：
    #   - 固定值：15.0
    #   - 范围：[min, max]，如 [14.0, 20.0]
    #   - 随机选择：[val1, val2, ...]，如 [0.6, 0.7, 0.8]
    #   - 相位值：以π为单位，[0, 2] 表示 [0, 2π]，[0, 0.15] 表示 [0, 0.15π]
    snr_db: [14.0, 20.0]  # SNR范围（dB），或固定值如 15.0
    amp_ratio: [0.2, 0.9]  # 幅度比范围，或固定值如 0.7
    freq_offset1: [0.0, 200.0]  # 第一路频偏范围（Hz）
    freq_offset2: [0.0, 200.0]  # 第二路频偏范围（Hz）
    phase1: [0.0, 0.15]  # 第一路初相位范围（以π为单位）
    phase2: [0.0, 2]  # 第二路初相位范围（以π为单位），[0, 2] 表示 [0, 2π]
    delay1_samp: [0, 8]  # 第一路时延范围（采样点）
    delay2_samp: [0, 8]  # 第二路时延范围（采样点）
    filter_type: rrc  # "rc" 或 "rrc"
```

**生成的文件名格式：**
```
{调制1}-{调制2}_snr{SNR}_amp{幅度比}_f1{频偏1}_f2{频偏2}_phi1{相位1}_phi2{相位2}_d1{时延1}_d2{时延2}_{滤波器}_N{样本数}_shard{分片号}_of{总分片数}_{数据类型}.pth
```

**示例文件名：**
```
8PSK-8PSK_snr14-20_amp0.2-0.9_f10-200_f20-200_phi10-0.15pi_phi20-2pi_d10-8_d20-8_RRC_N100000_shard01_of10_c128.pth
```

---

### Step 0B. 真实数据生成（从原始数据生成训练数据集）

数据生成分为两个步骤：

#### Step 0.1. 从原始数据切片并评估可解调性
```bash
python generate_data/split_from_raw_data.py --config configs/base_config.yaml
```

**功能说明：**
- 从原始 `.dat` 文件读取信号数据
- 切片并评估每个切片是否可解调（使用聚类分数阈值）
- 将切片分为训练集和测试集
- 输出切片文件：`{output_dir}/{modulation}/train_slices.npy` 和 `test_slices.npy`

**配置示例（在 `configs/base_config.yaml` 中）：**
```yaml
data_generation:
  split:
    modulation_list: [8PSK]  # 要处理的调制方式
    output_dir: /nas/datasets/yixin/PCMA/real_data  # 输出目录
    threshold: 6.0  # 聚类分数阈值（判断是否可解调）
    train_ratio: 0.9  # 训练集比例
    max_samples: 1000000  # 最多生成的样本数
    num_workers: 32  # 并行进程数
```

#### Step 0.2. 从切片生成混合信号对
```bash
python generate_data/generate_mixed_from_splits.py --config configs/base_config.yaml
```

**功能说明：**
- 从切片文件中读取训练集和测试集切片
- 生成混合信号对（两两加和，随机幅度比）
- 可选：添加噪声至目标SNR
- 分shard保存，每个shard包含指定数量的样本
- 输出文件：`{output_dir}/train/real_8psk_mixed_amp0.7_shard01_of10_c128.pth` 等

**配置示例（在 `configs/base_config.yaml` 中）：**
```yaml
data_generation:
  generate_mixed:
    modulation: 8PSK  # 调制方式
    mode: both  # "both"（训练+测试）, "train"（仅训练）, "test"（仅测试）
    output_dir: /nas/datasets/yixin/PCMA/real_data/8psk  # 输出目录
    shard_size: 10000  # 每个shard的样本数
    target_pairs: 100000  # 训练集目标生成的混合信号对数量
    amp_range: [0.7, 0.7]  # 幅度比范围，[0.7, 0.7] 表示固定值0.7
    # amp_list: [0.6, 0.7, 0.8, 0.9]  # 或使用列表，为每个幅度比生成单独文件
    add_noise_to_target_snr: false  # 是否添加噪声
    target_snr_db: 15.0  # 目标SNR（如果添加噪声）
```

**生成的数据文件结构：**
```
/nas/datasets/yixin/PCMA/real_data/8psk/
├── train/
│   ├── real_8psk_mixed_amp0.7_shard01_of10_c128.pth
│   ├── real_8psk_mixed_amp0.7_shard02_of10_c128.pth
│   └── ...
└── test/
    ├── real_8psk_mixed_amp0.7_shard01_of01_c128.pth
    └── ...
```

### Step 1. 配置训练数据路径

在 `configs/base_config.yaml` 中配置训练和测试数据路径：

**方式A：直接指定文件路径列表**
```yaml
data:
  signal_len: 3072  # 信号长度
  modulation: 8PSK  # 调制方式（用于解调）
  train:
    paths:
      - /nas/datasets/yixin/PCMA/real_data/8psk/train/real_8psk_mixed_amp0.7_shard01_of10_c128.pth
      - /nas/datasets/yixin/PCMA/real_data/8psk/train/real_8psk_mixed_amp0.7_shard02_of10_c128.pth
      # ... 添加所有训练shard文件
  test:
    paths:
      - /nas/datasets/yixin/PCMA/real_data/8psk/test/real_8psk_mixed_amp0.7_shard01_of01_c128.pth
```

**方式B：使用 base + shard_list + pattern（推荐，更简洁）**
```yaml
data:
  signal_len: 3072
  modulation: 8PSK
  train:
    base: /nas/datasets/yixin/PCMA/real_data/8psk/train
    shard_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # shard编号列表
    pattern: "real_8psk_mixed_amp0.7_shard{idx:02d}_of10_c128.pth"
  test:
    base: /nas/datasets/yixin/PCMA/real_data/8psk/test
    shard_list: [1]
    pattern: "real_8psk_mixed_amp0.7_shard{idx:02d}_of01_c128.pth"
```

### Step 2. 训练模型
```bash
python train_PCMA_diffusion.py --config base_config.yaml
```

**训练配置示例：**
```yaml
training:
  train_batch_size: 64  # 根据显存调整
  test_batch_size: 64
  num_epochs: 500
  learning_rate: 2e-4
  output_dir: results/DDPM-PCMA-8PSK  # 模型保存路径
```

### Step 3. 运行推理并保存输出
```bash
python generate_diffusion_output.py --config base_config.yaml
```

**采样配置示例：**
```yaml
sampling:
  num_inference_steps: 100  # 采样步数（100~500）
  eta: 0.15  # DDIM超参数
  output_dir: /nas/datasets/yixin/PCMA/8psk/diffusion_prediction/  # 输出目录
```

### Step 4. 解码（解调）
#### 仿真数据
```bash
python test_sim_decoder.py --config base_config.yaml
```
#### 真实数据
```bash
python test_decoder_ser_multi_process.py --config base_config.yaml
```

---

## 完整示例：生成 8PSK 数据并训练

### 1. 修改配置文件 `configs/base_config.yaml`

```yaml
# 数据生成配置
data_generation:
  raw_data:
    base_dir: /nas/datasets/LYX/PCMA
    paths:
      8PSK: /nas/datasets/LYX/PCMA/8PSK_16/1050/1050000000_20000000__24000000_20250623162534_672204_0000.dat
  
  split:
    modulation_list: [8PSK]
    output_dir: /nas/datasets/yixin/PCMA/real_data
    threshold: 6.0
    train_ratio: 0.9
    max_samples: 1000000
    num_workers: 32
  
  generate_mixed:
    modulation: 8PSK
    mode: both
    output_dir: /nas/datasets/yixin/PCMA/real_data/8psk
    shard_size: 10000
    target_pairs: 100000  # 生成10万组训练数据
    amp_range: [0.7, 0.7]
    add_noise_to_target_snr: false

# 训练数据路径配置
data:
  signal_len: 3072
  modulation: 8PSK
  train:
    base: /nas/datasets/yixin/PCMA/real_data/8psk/train
    shard_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pattern: "real_8psk_mixed_amp0.7_shard{idx:02d}_of10_c128.pth"
  test:
    base: /nas/datasets/yixin/PCMA/real_data/8psk/test
    shard_list: [1]
    pattern: "real_8psk_mixed_amp0.7_shard{idx:02d}_of01_c128.pth"

# 训练配置
training:
  train_batch_size: 64
  num_epochs: 500
  output_dir: results/DDPM-PCMA-8PSK
```

### 2. 执行数据生成
```bash
# Step 1: 切片并评估
python generate_data/split_from_raw_data.py --config configs/base_config.yaml

# Step 2: 生成混合信号对
python generate_data/generate_mixed_from_splits.py --config configs/base_config.yaml
```

### 3. 训练模型
```bash
python train_PCMA_diffusion.py --config base_config.yaml
```

---

## 注意事项

1. **数据路径**：确保原始数据文件路径正确，且输出目录有写入权限
2. **内存和存储**：生成大量数据时注意磁盘空间和内存使用
3. **并行处理**：`num_workers` 根据CPU核心数调整，避免过多进程导致内存不足
4. **数据隔离**：训练集和测试集严格隔离，避免数据泄露
5. **文件命名**：生成的文件名包含关键参数信息，便于识别和管理
