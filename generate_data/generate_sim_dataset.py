import numpy as np
from scipy.signal import convolve
import torch
import os
import argparse
import yaml

# ============= 系统参数（通常不需要修改） =============
beta = 0.33          # 滚降系数
sps = 8              # 每符号采样点数
fs = 12e6            # 采样频率
num_taps = 64        # 滤波器抽头数
input_len = 3072     # 每块样本点数
assert input_len % sps == 0
num_syms = input_len // sps  # 每路符号数

# 各调制方式每符号 bit 数
BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

# 可选调制集合
MOD_LIST = ["QPSK", "8PSK", "16QAM"]


# ============= 配置加载函数 =============
def load_config_from_yaml(config_path):
    """
    从 YAML 文件加载配置
    
    参数设置方式（在 YAML 中）：
      1. 固定值：直接写数值，例如 15.0, 0.7, 0
      2. 范围（均匀分布）：列表 [min, max]，例如 [14.0, 20.0]
      3. 列表（随机选择）：列表 [val1, val2, ...]，例如 [0.6, 0.7, 0.8]
      4. 相位值：可以使用数值（rad），或使用 "pi" 表示 π，例如 [0, "pi"] 或 [0, 2] 表示 [0, 2π]
    
    注意：YAML 中的列表会被转换为元组或列表，相位值中的 "pi" 会被转换为 np.pi
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'data_generation' not in config:
        raise ValueError("YAML 配置文件中缺少 'data_generation' 部分")
    
    sim_cfg = config['data_generation'].get('generate_sim', {})
    
    def convert_value(val):
        """转换配置值：处理列表、元组、相位值等"""
        if isinstance(val, list):
            # 检查是否包含 "pi" 字符串（相位值）
            if any(isinstance(v, str) and v.lower() == "pi" for v in val):
                # 将 "pi" 转换为 np.pi
                converted = []
                for v in val:
                    if isinstance(v, str) and v.lower() == "pi":
                        converted.append(np.pi)
                    else:
                        converted.append(float(v))
                # 如果是两个元素的列表，转换为元组（表示范围）
                if len(converted) == 2:
                    return tuple(converted)
                return converted
            # 普通列表：如果只有两个元素，可能是范围，转换为元组
            if len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
                return tuple(val)
            return val
        elif isinstance(val, (int, float)):
            return val
        elif isinstance(val, str):
            # 字符串可能是调制方式
            return val
        else:
            return val
    
    # 构建配置字典
    config_dict = {
        'num_samples': sim_cfg.get('num_samples', 1000),
        'shard_size': sim_cfg.get('shard_size', 0),
        'save_dir': sim_cfg.get('save_dir', '/nas/datasets/yixin/PCMA/temp'),
        'save_complex64': sim_cfg.get('save_complex64', True),
        'random_seed': sim_cfg.get('random_seed'),
        'modulation1': sim_cfg.get('modulation1', '8PSK'),
        'modulation2': sim_cfg.get('modulation2', '8PSK'),
        'snr_db': convert_value(sim_cfg.get('snr_db', [14.0, 20.0])),
        'amp_ratio': convert_value(sim_cfg.get('amp_ratio', [0.2, 0.9])),
        'freq_offset1': convert_value(sim_cfg.get('freq_offset1', [0.0, 200.0])),
        'freq_offset2': convert_value(sim_cfg.get('freq_offset2', [0.0, 200.0])),
        'phase1': convert_value(sim_cfg.get('phase1', [0.0, 0.15])),  # 默认 [0, 0.15π]
        'phase2': convert_value(sim_cfg.get('phase2', [0.0, 2])),    # 默认 [0, 2π]，使用 "pi" 或数值
        'delay1_samp': convert_value(sim_cfg.get('delay1_samp', [0, sps])),
        'delay2_samp': convert_value(sim_cfg.get('delay2_samp', [0, sps])),
        'filter_type': sim_cfg.get('filter_type', 'rrc'),
    }
    
    # 处理相位值：如果 phase1/phase2 是列表且包含数值，需要检查是否需要乘以 π
    # YAML 中可以用 [0, 2] 表示 [0, 2π]，或者 [0, "pi"] 表示 [0, π]
    for phase_key in ['phase1', 'phase2']:
        phase_val = config_dict[phase_key]
        if isinstance(phase_val, (list, tuple)) and len(phase_val) == 2:
            # 检查原始配置，看是否使用了 "pi" 标记
            orig_val = sim_cfg.get(phase_key)
            if isinstance(orig_val, list):
                # 如果原始配置中有 "pi" 字符串，已经转换过了
                # 如果没有，且值较小（< 10），可能是以 π 为单位的值
                if not any(isinstance(v, str) and v.lower() == "pi" for v in orig_val):
                    # 检查值是否看起来像是以 π 为单位（例如 0.15, 2 等）
                    if all(isinstance(v, (int, float)) and v < 10 for v in orig_val):
                        # 假设是以 π 为单位，转换为弧度
                        config_dict[phase_key] = tuple(v * np.pi if v != 0 else 0.0 for v in phase_val)
        elif isinstance(phase_val, (int, float)) and phase_val < 10:
            # 单个值，如果小于10，可能是以 π 为单位
            orig_val = sim_cfg.get(phase_key)
            if isinstance(orig_val, (int, float)) and orig_val < 10:
                config_dict[phase_key] = phase_val * np.pi if phase_val != 0 else 0.0
    
    return config_dict


# ============= 辅助函数 =============
def get_bit_len(modulation: str) -> int:
    """给定调制方式，返回每路比特长度。"""
    return num_syms * BITS_PER_SYMBOL[modulation.upper()]


def sample_param(param_config):
    """
    从参数配置中采样一个值。
    支持：
    - 固定值：直接返回
    - 范围 (min, max)：均匀分布采样
    - 列表 [val1, val2, ...]：随机选择
    - numpy数组：随机选择
    """
    if isinstance(param_config, (int, float)):
        return param_config
    elif isinstance(param_config, tuple) and len(param_config) == 2:
        return np.random.uniform(param_config[0], param_config[1])
    elif isinstance(param_config, (list, np.ndarray)):
        return np.random.choice(param_config)
    else:
        raise ValueError(f"不支持的参数配置类型: {type(param_config)}")


def sample_modulation(mod_config):
    """采样调制方式。"""
    if isinstance(mod_config, str):
        return mod_config.upper()
    elif isinstance(mod_config, (list, np.ndarray)):
        return np.random.choice(mod_config).upper()
    else:
        raise ValueError(f"不支持的调制配置类型: {type(mod_config)}")


# ============= 调制函数 =============
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """QPSK Gray 映射。"""
    symbols = []
    for i in range(0, len(bits), 2):
        b1, b2 = bits[i], bits[i + 1]
        if b1 == 0 and b2 == 0:
            symbols.append(1 + 1j)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1 + 1j)
        elif b1 == 1 and b2 == 0:
            symbols.append(1 - 1j)
        else:
            symbols.append(-1 - 1j)
    return np.array(symbols, dtype=complex) / np.sqrt(2)


def psk8_mod(bits: np.ndarray) -> np.ndarray:
    """8PSK 映射：每 3bit -> 一个符号，自然编码。"""
    assert len(bits) % 3 == 0
    bits = bits.reshape(-1, 3)
    idx = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    phase = 2 * np.pi * idx / 8.0
    symbols = np.exp(1j * phase)
    return symbols.astype(complex)


def qam16_mod(bits: np.ndarray) -> np.ndarray:
    """16QAM 映射：每 4bit -> 2bit(I) + 2bit(Q)，Gray编码。"""
    assert len(bits) % 4 == 0
    bits = bits.reshape(-1, 4)
    
    def gray2level(b0, b1):
        if b0 == 0 and b1 == 0:
            return -3
        elif b0 == 0 and b1 == 1:
            return -1
        elif b0 == 1 and b1 == 1:
            return 1
        else:
            return 3

    I = np.array([gray2level(b[0], b[1]) for b in bits], dtype=float)
    Q = np.array([gray2level(b[2], b[3]) for b in bits], dtype=float)
    symbols = I + 1j * Q
    symbols = symbols / np.sqrt(10.0)  # 平均能量归一化
    return symbols.astype(complex)


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """统一入口：根据 modulation 调用对应的调制函数。"""
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_mod(bits)
    elif modulation == "8PSK":
        return psk8_mod(bits)
    elif modulation == "16QAM":
        return qam16_mod(bits)
    else:
        raise ValueError(f"不支持的调制方式: {modulation}")


# ============= 滤波器 =============
def rc_filter(beta, sps, num_taps):
    """RC (Raised Cosine) 滤波器。"""
    t = np.arange(-num_taps // 2, num_taps // 2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def rrc_filter(beta, sps, num_taps):
    """RRC (Root-Raised-Cosine) 滤波器。"""
    t = np.arange(-num_taps // 2, num_taps // 2, dtype=np.float64) / float(sps)
    Ts = 1.0
    beta = float(beta)

    h = np.zeros_like(t, dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti / Ts) - 1.0) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - beta) / Ts)
                + 4 * beta * ti / Ts * np.cos(np.pi * ti * (1 + beta) / Ts)
            )
            den = np.pi * ti / Ts * (1 - (4 * beta * ti / Ts) ** 2)
            h[i] = num / den

    h = h / np.sqrt(np.sum(h ** 2))
    return h


# 预计算滤波器
rc = rc_filter(beta, sps, num_taps)
rrc = rrc_filter(beta, sps, num_taps)


# ============= 噪声 =============
def awgn_with_seed(signal, snr_db, seed=None):
    """添加AWGN噪声。"""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


# ============= 数据归一化 =============
def energy_normalize_dataset(dataset):
    """能量归一化数据集。"""
    energies = [np.mean(np.abs(e['mixsignal']) ** 2) for e in dataset]
    mean_e = np.mean(energies) if energies else 1.0
    scale = np.sqrt(mean_e)
    for e in dataset:
        e['mixsignal'] /= scale
        e['rfsignal1'] /= scale
        e['rfsignal2'] /= scale
    return dataset


def maybe_cast_complex64(entry, save_complex64):
    """如果需要，转换为complex64。"""
    if save_complex64:
        entry['mixsignal'] = entry['mixsignal'].astype(np.complex64)
        entry['rfsignal1'] = entry['rfsignal1'].astype(np.complex64)
        entry['rfsignal2'] = entry['rfsignal2'].astype(np.complex64)
    return entry


# ============= 核心生成函数 =============
def generate_one_sample(config):
    """
    生成一个样本。
    
    返回字典，包含：
    - mixsignal: 混合信号
    - rfsignal1: 第一路信号
    - rfsignal2: 第二路信号
    - params: 参数元组
    - bits1: 第一路比特
    - bits2: 第二路比特
    """
    # 1) 采样调制方式
    mod1 = sample_modulation(config["modulation1"])
    mod2 = sample_modulation(config["modulation2"])
    
    # 2) 生成随机比特并调制
    bit_len1 = get_bit_len(mod1)
    bit_len2 = get_bit_len(mod2)
    bits1 = np.random.randint(0, 2, bit_len1, dtype=np.int8)
    bits2 = np.random.randint(0, 2, bit_len2, dtype=np.int8)
    symbols1 = modulate(bits1, mod1)
    symbols2 = modulate(bits2, mod2)

    assert len(symbols1) == num_syms
    assert len(symbols2) == num_syms

    # 3) 采样参数
    snr_db = sample_param(config["snr_db"])
    amp_ratio = sample_param(config["amp_ratio"])
    freq_off1 = sample_param(config["freq_offset1"])
    freq_off2 = sample_param(config["freq_offset2"])
    phi1 = sample_param(config["phase1"])
    phi2 = sample_param(config["phase2"])
    delay1 = int(sample_param(config["delay1_samp"]))
    delay2 = int(sample_param(config["delay2_samp"]))
    
    # 注意：如果需要符号随机化，请配置范围如 (-200.0, 200.0)
    
    # 4) 上采样 + 时延
    up_len = num_syms * sps
    symbols_up1 = np.zeros(up_len, dtype=complex)
    symbols_up2 = np.zeros(up_len, dtype=complex)
    symbols_up1[delay1::sps] = symbols1
    symbols_up2[delay2::sps] = symbols2 * amp_ratio
    
    # 5) 成型滤波
    filter_type = config.get("filter_type", "rrc").lower()
    filter_h = rrc if filter_type == "rrc" else rc
    tx1 = convolve(symbols_up1, filter_h, mode='same')
    tx2 = convolve(symbols_up2, filter_h, mode='same')
    
    # 6) CFO + 初相位
    t = np.arange(up_len) / fs
    tx1 = tx1 * np.exp(1j * (2 * np.pi * freq_off1 * t + phi1))
    tx2 = tx2 * np.exp(1j * (2 * np.pi * freq_off2 * t + phi2))

    # 7) 合路 + AWGN
    rx_clean = tx1 + tx2
    rx = awgn_with_seed(rx_clean, snr_db, seed=None)

    # 8) 构建样本字典
    entry = {
            'mixsignal': rx,
            'rfsignal1': tx1,
            'rfsignal2': tx2,
            'params': (
                float(snr_db), float(amp_ratio), sps,
            f'f_off1={float(freq_off1):.2f}Hz',
            f'f_off2={float(freq_off2):.2f}Hz',
                f'phi1={float(phi1):.4f}rad',
                f'phi2={float(phi2):.4f}rad',
            f'delay1_samp={delay1}',
            f'delay2_samp={delay2}',
            f'mod1={mod1}',
            f'mod2={mod2}',
            ),
            'bits1': bits1,
            'bits2': bits2,
        'origin_len': 1
    }
    
    return entry


# ============= 数据集生成和保存 =============
def generate_dataset(config):
    """
    根据配置生成完整数据集并保存。
    """
    # 设置随机种子
    if config.get("random_seed") is not None:
        np.random.seed(config["random_seed"])
    
    num_samples = config["num_samples"]
    shard_size = config.get("shard_size", 0)
    save_dir = config["save_dir"]
    save_complex64 = config.get("save_complex64", False)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 确定是否分片
    if shard_size > 0 and shard_size < num_samples:
        num_shards = (num_samples + shard_size - 1) // shard_size
        use_sharding = True
    else:
        num_shards = 1
        use_sharding = False
        shard_size = num_samples
    
    print(f"开始生成数据集...")
    print(f"  总样本数: {num_samples}")
    print(f"  分片大小: {shard_size if use_sharding else '不分片'}")
    print(f"  保存目录: {save_dir}")
    print(f"  数据类型: {'complex64' if save_complex64 else 'complex128'}")
    
    shard_entries = []
    shard_idx = 1
    saved_paths = []
    
    def get_config_tag(key, default="Var", is_phase=False):
        """
        从配置中获取参数标签用于文件名。
        
        Args:
            key: 配置键名
            default: 默认值
            is_phase: 是否为相位参数（需要除以π）
        """
        val = config.get(key)
        if isinstance(val, (int, float)):
            if is_phase:
                # 相位值除以π
                val_pi = val / np.pi
                # 如果接近0，显示为0
                if abs(val_pi) < 1e-6:
                    return "0"
                # 如果是整数倍π，显示为整数+pi
                elif abs(val_pi - round(val_pi)) < 1e-6:
                    pi_mult = int(round(val_pi))
                    if pi_mult == 1:
                        return "pi"
                    elif pi_mult == -1:
                        return "-pi"
                    else:
                        return f"{pi_mult}pi"
                else:
                    return f"{val_pi:.2f}pi"
            else:
                # 非相位参数
                if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                    return f"{int(val)}"
                else:
                    return f"{val:.1f}"
        elif isinstance(val, tuple) and len(val) == 2:
            # 范围：显示为 min-max
            v1, v2 = val[0], val[1]
            if is_phase:
                # 相位值都除以π
                v1_pi = v1 / np.pi
                v2_pi = v2 / np.pi
                
                # 格式化v1
                if abs(v1_pi) < 1e-6:
                    v1_str = "0"
                elif abs(v1_pi - round(v1_pi)) < 1e-6:
                    pi_mult = int(round(v1_pi))
                    if pi_mult == 1:
                        v1_str = "pi"
                    elif pi_mult == -1:
                        v1_str = "-pi"
                    else:
                        v1_str = f"{pi_mult}pi"
                else:
                    v1_str = f"{v1_pi:.2f}pi"
                
                # 格式化v2
                if abs(v2_pi) < 1e-6:
                    v2_str = "0"
                elif abs(v2_pi - round(v2_pi)) < 1e-6:
                    pi_mult = int(round(v2_pi))
                    if pi_mult == 1:
                        v2_str = "pi"
                    elif pi_mult == -1:
                        v2_str = "-pi"
                    else:
                        v2_str = f"{pi_mult}pi"
                else:
                    v2_str = f"{v2_pi:.2f}pi"
                
                return f"{v1_str}-{v2_str}"
            else:
                # 非相位参数
                if isinstance(v1, int) or (isinstance(v1, float) and v1.is_integer()):
                    v1_str = f"{int(v1)}"
                else:
                    v1_str = f"{v1:.1f}"
                if isinstance(v2, int) or (isinstance(v2, float) and v2.is_integer()):
                    v2_str = f"{int(v2)}"
                else:
                    v2_str = f"{v2:.1f}"
                return f"{v1_str}-{v2_str}"
        elif isinstance(val, str):
            return val.upper()
        elif isinstance(val, (list, np.ndarray)):
            return "Mixed"
        else:
            return default
    
    def flush_shard(entries, idx):
        """保存当前分片。"""
        if not entries:
            return None
        
        entries_norm = energy_normalize_dataset(entries)
        entries_norm = [maybe_cast_complex64(e, save_complex64) for e in entries_norm]
        
        # 构建文件名（包含所有关键参数信息）
        dtype_tag = "_c64" if save_complex64 else "_c128"
        
        # 获取调制方式标签
        mod1_tag = get_config_tag("modulation1", "Mixed")
        mod2_tag = get_config_tag("modulation2", "Mixed")
        mod_tag = f"{mod1_tag}-{mod2_tag}"
        
        # 获取各个参数标签
        snr_tag = f"snr{get_config_tag('snr_db', 'Var')}"
        amp_tag = f"amp{get_config_tag('amp_ratio', 'Var')}"
        f1_tag = f"f1{get_config_tag('freq_offset1', 'Var')}"
        f2_tag = f"f2{get_config_tag('freq_offset2', 'Var')}"
        phi1_tag = f"phi1{get_config_tag('phase1', 'Var', is_phase=True)}"
        phi2_tag = f"phi2{get_config_tag('phase2', 'Var', is_phase=True)}"
        d1_tag = f"d1{get_config_tag('delay1_samp', 'Var')}"
        d2_tag = f"d2{get_config_tag('delay2_samp', 'Var')}"
        filter_tag = config.get("filter_type", "rrc").upper()
        
        # 构建基础文件名（按逻辑分组）
        # 格式：{调制}_{SNR}_{amp}_{频偏}_{相位}_{时延}_{滤波器}_N{样本数}
        param_parts = [
            mod_tag,
            snr_tag,
            amp_tag,
            f"{f1_tag}_{f2_tag}",
            f"{phi1_tag}_{phi2_tag}",
            f"{d1_tag}_{d2_tag}",
            filter_tag,
            f"N{num_samples}"
        ]
        base_name = "_".join(param_parts)
        
        if use_sharding:
            filename = f"{base_name}_shard{idx:02d}_of{num_shards:02d}{dtype_tag}.pth"
        else:
            filename = f"{base_name}{dtype_tag}.pth"
        
        path = os.path.join(save_dir, filename)
        torch.save(entries_norm, path)
        print(f"📦 已保存分片 {idx}/{num_shards}: {path} （样本数 {len(entries_norm)}）")
        return path
    
    # 生成样本
    for k in range(num_samples):
        entry = generate_one_sample(config)
        shard_entries.append(entry)

        # 进度打印
        if (k + 1) % 1000 == 0 or (k + 1) == num_samples:
            print(f"进度 {k + 1}/{num_samples} ({100.0*(k+1)/num_samples:.1f}%)")

        # 分片保存
        if use_sharding and len(shard_entries) >= shard_size:
            p = flush_shard(shard_entries, shard_idx)
            if p:
                saved_paths.append(p)
            shard_entries = []
            shard_idx += 1

    # 保存最后一个分片
    if shard_entries:
        p = flush_shard(shard_entries, shard_idx)
        if p:
            saved_paths.append(p)
    
    print(f"\n✅ 数据集生成完成！")
    print(f"  总样本数: {num_samples}")
    print(f"  分片数: {len(saved_paths)}")
    if saved_paths:
        print(f"  示例路径: {saved_paths[0]}")
    
    return saved_paths


# ============= 主函数 =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成仿真数据集')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='配置文件路径（默认: configs/base_config.yaml）')
    args = parser.parse_args()
    
    # 从 YAML 加载配置
    config = load_config_from_yaml(args.config)
    
    print(f"配置文件: {args.config}")
    print(f"保存目录: {config['save_dir']}")
    print(f"总样本数: {config['num_samples']}")
    print(f"分片大小: {config['shard_size'] if config['shard_size'] > 0 else '不分片'}")
    print(f"{'='*60}\n")
    
    # 生成数据集
    generate_dataset(config)
