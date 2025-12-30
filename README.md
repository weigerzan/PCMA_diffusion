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
通常只需修改数据路径。

---

## Usage

### Step 1. Train the Model
```bash
python train_PCMA_diffusion.py --config base_config.yaml
```

### Step 2. Run Inference and Save Outputs
```bash
python generate_diffusion_output.py --config base_config.yaml
```

### Step 3. Decode (Demodulation)
#### 仿真数据
```bash
python test_sim_decoder.py --config base_config.yaml
```
#### 真实数据
```bash
python test_decoder_ser_multi_process.py --config base_config.yaml
```
