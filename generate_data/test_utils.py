"""
测试工具函数模块
统一管理信号生成、调制、滤波等通用函数
"""

import numpy as np
from scipy.signal import convolve
from typing import Tuple, Optional, Dict
from scipy import signal as signal_module
# ==================== 常量定义 ====================
BETA = 0.33
SPS = 8
FS = 12e6
SYMBOL_RATE = FS / SPS  # 符号率 = 采样率 / 每符号采样数 = 1.5e6 symbols/sec
NUM_TAPS = 64
INPUT_LEN = 3072
NUM_SYMS = INPUT_LEN // SPS
TARGET_SPS = 8
SLICE_LENGTH = 3072
BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

MOD_LIST = ["QPSK", "8PSK", "16QAM"]


# ==================== 调制函数 ====================
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """
    QPSK Gray 映射
    
    Args:
        bits: 比特序列，长度必须是2的倍数
        
    Returns:
        符号序列
    """
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
    """
    8PSK 映射：每 3bit -> 一个符号，采用自然编码
    
    Args:
        bits: 比特序列，长度必须是3的倍数
        
    Returns:
        符号序列
    """
    assert len(bits) % 3 == 0, f"比特长度必须是3的倍数，当前长度: {len(bits)}"
    bits = bits.reshape(-1, 3)
    idx = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    phase = 2 * np.pi * idx / 8.0
    symbols = np.exp(1j * phase)
    return symbols.astype(complex)


def qam16_mod(bits: np.ndarray) -> np.ndarray:
    """
    16QAM 映射：每 4bit -> 2bit(I) + 2bit(Q)，采用 Gray 编码
    
    Args:
        bits: 比特序列，长度必须是4的倍数
        
    Returns:
        符号序列
    """
    assert len(bits) % 4 == 0, f"比特长度必须是4的倍数，当前长度: {len(bits)}"
    bits = bits.reshape(-1, 4)
    
    # Gray 2bit -> level 映射
    def gray2level(b0, b1):
        if b0 == 0 and b1 == 0:
            return -3
        elif b0 == 0 and b1 == 1:
            return -1
        elif b0 == 1 and b1 == 1:
            return 1
        else:  # b0==1 and b1==0
            return 3
    
    I = np.array([gray2level(b[0], b[1]) for b in bits], dtype=float)
    Q = np.array([gray2level(b[2], b[3]) for b in bits], dtype=float)
    symbols = I + 1j * Q
    symbols = symbols / np.sqrt(10.0)  # 平均能量归一化
    return symbols.astype(complex)


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    统一入口：根据 modulation 调用对应的调制函数
    
    Args:
        bits: 比特序列
        modulation: 调制方式 ("QPSK", "8PSK", "16QAM")
        
    Returns:
        符号序列
    """
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_mod(bits)
    elif modulation == "8PSK":
        return psk8_mod(bits)
    elif modulation == "16QAM":
        return qam16_mod(bits)
    else:
        raise ValueError(f"不支持的调制方式: {modulation}。有效值: {MOD_LIST}")


# ==================== 滤波器函数 ====================
def rc_filter(beta: float, sps: int, num_taps: int) -> np.ndarray:
    """
    创建 RC（Raised-Cosine）滤波器
    
    Args:
        beta: 滚降因子
        sps: 每符号采样数
        num_taps: 滤波器抽头数
        
    Returns:
        滤波器系数
    """
    t = np.arange(-num_taps // 2, num_taps // 2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def rrc_filter(beta: float, sps: int, num_taps: int) -> np.ndarray:
    """
    创建 RRC（Root-Raised-Cosine）滤波器
    
    Args:
        beta: 滚降因子
        sps: 每符号采样数
        num_taps: 滤波器抽头数
        
    Returns:
        滤波器系数
    """
    t = np.arange(-num_taps // 2, num_taps // 2, dtype=np.float64) / float(sps)
    Ts = 1.0
    beta = float(beta)
    
    h = np.zeros_like(t, dtype=np.float64)
    
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            # t = 0
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti / Ts) - 1.0) < 1e-8:
            # t = ± Ts/(4β)
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
    
    # 单位能量归一化
    h = h / np.sqrt(np.sum(h ** 2))
    return h


# ==================== 噪声函数 ====================
def awgn_with_seed(signal: np.ndarray, snr_db: float, seed: Optional[int] = None) -> np.ndarray:
    """
    添加AWGN噪声
    
    Args:
        signal: 输入信号
        snr_db: 信噪比（dB）
        seed: 随机种子（可选）
        
    Returns:
        添加噪声后的信号
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        )
    else:
        noise = np.sqrt(noise_power / 2) * (
            np.random.standard_normal(len(signal)) + 1j * np.random.standard_normal(len(signal))
        )
    return signal + noise


# ==================== 信号生成函数 ====================
def generate_signal_pair(
    bits1: np.ndarray,
    bits2: np.ndarray,
    modulation1: str,
    modulation2: str,
    amp_ratio: float = 1.0,
    freq_offset1_hz: float = 0.0,
    freq_offset2_hz: float = 0.0,
    phase1_rad: float = 0.0,
    phase2_rad: float = 0.0,
    delay1_samp: int = 0,
    delay2_samp: int = 0,
    snr_db: Optional[float] = None,
    seed: Optional[int] = None,
    sps: int = SPS,
    fs: float = FS,
    beta: float = BETA,
    num_taps: int = NUM_TAPS,
    use_rrc: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成两路信号并混合（与generate_sim_dataset.py一致）
    
    Args:
        bits1, bits2: 比特序列
        modulation1, modulation2: 调制方式
        amp_ratio: 信号2相对于信号1的幅度比
        freq_offset1_hz, freq_offset2_hz: 频偏（Hz）
        phase1_rad, phase2_rad: 相偏（弧度）
        delay1_samp, delay2_samp: 时延（采样点数）
        snr_db: 信噪比（针对混合信号，如果为None则不添加噪声）
        seed: 随机种子（用于噪声生成）
        sps: 每符号采样数
        fs: 采样频率（Hz）
        beta: RRC滚降因子
        num_taps: 滤波器抽头数
        use_rrc: 是否使用RRC滤波器（True使用RRC，False使用RC）
    
    Returns:
        (mixsignal, rfsignal1, rfsignal2, symbols1, symbols2)
        - mixsignal: 混合信号（已添加噪声）
        - rfsignal1: 信号1（未添加噪声）
        - rfsignal2: 信号2（未添加噪声）
        - symbols1, symbols2: 原始符号
    """
    # 调制
    symbols1 = modulate(bits1, modulation1)
    symbols2 = modulate(bits2, modulation2)
    
    # 上采样
    up_len = len(symbols1) * sps
    assert len(symbols1) == len(symbols2), "符号数必须相同"
    
    symbols_up1 = np.zeros(up_len, dtype=complex)
    symbols_up2 = np.zeros(up_len, dtype=complex)
    symbols_up1[delay1_samp::sps] = symbols1
    symbols_up2[delay2_samp::sps] = symbols2 * amp_ratio
    
    # 成型滤波
    if use_rrc:
        filter_coeff = rrc_filter(beta, sps, num_taps)
    else:
        filter_coeff = rc_filter(beta, sps, num_taps)
    
    tx1 = convolve(symbols_up1, filter_coeff, mode='same')
    tx2 = convolve(symbols_up2, filter_coeff, mode='same')
    
    # CFO和相位
    t = np.arange(up_len) / fs
    tx1 = tx1 * np.exp(1j * (2 * np.pi * freq_offset1_hz * t + phase1_rad))
    tx2 = tx2 * np.exp(1j * (2 * np.pi * freq_offset2_hz * t + phase2_rad))
    
    # 混合并添加噪声
    mixsignal_clean = tx1 + tx2
    if snr_db is not None:
        seed_rx = seed ^ 0x12345678 if seed is not None else None
        mixsignal = awgn_with_seed(mixsignal_clean, snr_db, seed_rx)
    else:
        mixsignal = mixsignal_clean
    
    return mixsignal, tx1, tx2, symbols1, symbols2


# ==================== 能量归一化 ====================
def energy_normalize_signal(mixsignal: np.ndarray, 
                           signal1: Optional[np.ndarray] = None,
                           signal2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    """
    对信号进行能量归一化
    
    Args:
        mixsignal: 混合信号
        signal1: 信号1（可选）
        signal2: 信号2（可选）
        
    Returns:
        归一化后的信号（顺序与输入相同）
    """
    mix_energy = np.mean(np.abs(mixsignal) ** 2)
    scale = np.sqrt(mix_energy)
    
    result = [mixsignal / scale]
    if signal1 is not None:
        result.append(signal1 / scale)
    if signal2 is not None:
        result.append(signal2 / scale)
    
    return tuple(result) if len(result) > 1 else result[0]


def get_bit_len(modulation: str, num_syms: int = NUM_SYMS) -> int:
    """
    根据调制方式和符号数计算比特长度
    
    Args:
        modulation: 调制方式
        num_syms: 符号数
        
    Returns:
        比特长度
    """
    return num_syms * BITS_PER_SYMBOL[modulation.upper()]


# ==================== 解调函数 ====================
def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """
    QPSK解调，与qpsk_mod对应：
      00 -> (+,+)
      01 -> (-,+)
      10 -> (+,-)
      11 -> (-,-)
    """
    bits = []
    sym = symbols * np.sqrt(2)
    for s in sym:
        if s.real >= 0 and s.imag >= 0:
            b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0:
            b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0:
            b1, b2 = 1, 0
        else:
            b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits, dtype=np.int8)


def psk8_demod(symbols: np.ndarray) -> np.ndarray:
    """
    8PSK解调，对应psk8_mod：
      bits -> k = b0*4 + b1*2 + b2
      s = exp(j * 2πk/8)
    从角度恢复k，再还原bits
    """
    angles = np.angle(symbols)
    angles = np.mod(angles, 2 * np.pi)
    step = 2 * np.pi / 8.0
    k = np.round(angles / step).astype(int) % 8  # 0..7
    bits = []
    for val in k:
        b0 = (val >> 2) & 1  # 权重 4
        b1 = (val >> 1) & 1  # 权重 2
        b2 = val & 1         # 权重 1
        bits.extend([b0, b1, b2])
    return np.array(bits, dtype=np.int8)


def qam16_demod(symbols: np.ndarray) -> np.ndarray:
    """
    16QAM解调，对应qam16_mod：
      I,Q ∈ {-3,-1,1,3}/sqrt(10)，Gray编码
    按最近邻找I/Q所在的level，再映射回bits
    """
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10.0)
    bits = []
    for s in symbols:
        I = s.real
        Q = s.imag
        idx_I = np.argmin((I - levels) ** 2)
        idx_Q = np.argmin((Q - levels) ** 2)
        level_I = levels[idx_I]
        level_Q = levels[idx_Q]

        # level -> Gray bits
        if level_I < (-2 / np.sqrt(10)):
            bi0, bi1 = 0, 0   # -3
        elif level_I < 0:
            bi0, bi1 = 0, 1   # -1
        elif level_I > (2 / np.sqrt(10)):
            bi0, bi1 = 1, 0   # 3
        else:
            bi0, bi1 = 1, 1   # 1

        if level_Q < (-2 / np.sqrt(10)):
            bq0, bq1 = 0, 0
        elif level_Q < 0:
            bq0, bq1 = 0, 1
        elif level_Q > (2 / np.sqrt(10)):
            bq0, bq1 = 1, 0
        else:
            bq0, bq1 = 1, 1

        bits.extend([bi0, bi1, bq0, bq1])
    return np.array(bits, dtype=np.int8)


def demod_by_mod(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    根据调制方式解调符号
    
    Args:
        symbols: 符号序列
        modulation: 调制方式
        
    Returns:
        比特序列
    """
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_demod(symbols)
    elif modulation == "8PSK":
        return psk8_demod(symbols)
    elif modulation == "16QAM":
        return qam16_demod(symbols)
    else:
        # 默认当QPSK处理
        return qpsk_demod(symbols)


# ==================== 匹配滤波和符号抽取 ====================
def find_best_offset(y_mf: np.ndarray, sps: int) -> int:
    """
    找到最佳符号抽取offset
    
    Args:
        y_mf: 匹配滤波后的信号
        sps: 每符号采样数
        
    Returns:
        最佳offset
    """
    best_off = 0
    best_eng = -1.0
    for off in range(sps):
        sym = y_mf[off::sps]
        if len(sym) > 0:
            eng = np.mean(np.abs(sym) ** 2)
            if eng > best_eng:
                best_eng = eng
                best_off = off
    return best_off


def mf_and_sample(wave: np.ndarray, sps: int, matched_filter: np.ndarray, 
                  num_taps: int, guard_sym: Optional[int] = None,
                  filter_type: Optional[str] = None) -> np.ndarray:
    """
    匹配滤波 + 符号抽取
    
    Args:
        wave: 输入信号
        sps: 每符号采样数
        matched_filter: 匹配滤波器系数（RC或RRC）
        num_taps: 滤波器抽头数
        guard_sym: guard符号数（如果为None，则使用num_taps//sps）
        filter_type: 滤波器类型，'RC' 或 'RRC'。如果为None，则通过比较滤波器系数自动判断
        
    Returns:
        抽取的符号序列
    """
    if guard_sym is None:
        guard_sym = num_taps // sps  # 64/8=8 符号

    if wave is None or len(wave) == 0:
        return np.zeros(0, dtype=np.complex64)

    # 判断滤波器类型
    if filter_type is None:
        # 自动判断：生成一个RRC滤波器，比较系数来判断
        # 如果matched_filter是RRC，则进行匹配滤波；如果是RC，则跳过匹配滤波
        rrc_ref = rrc_filter(BETA, sps, num_taps)
        # 使用均方误差来判断是否相似（允许小的数值误差）
        mse = np.mean((matched_filter - rrc_ref) ** 2)
        is_rrc = mse < 1e-6  # 如果MSE很小，认为是RRC
    else:
        is_rrc = (filter_type.upper() == 'RRC')
    
    # RC滤波器：不进行匹配滤波，直接使用原始信号
    # RRC滤波器：进行匹配滤波
    if is_rrc:
        # RRC滤波器：需要匹配滤波
        try:
            y_mf = convolve(wave, matched_filter, mode='same')
        except Exception:
            # 如果失败，直接使用原始信号
            y_mf = wave
    else:
        # RC滤波器：不进行匹配滤波，直接使用原始信号
        y_mf = wave
    
    off = find_best_offset(y_mf, sps)
    syms = y_mf[off::sps]
    if len(syms) <= 2 * guard_sym:
        return np.zeros(0, dtype=np.complex64)
    syms = syms[guard_sym:-guard_sym]

    # 归一化
    m = np.mean(np.abs(syms))
    if m > 0:
        syms = syms / m
    return syms.astype(np.complex64)


# ==================== 比特对齐和SER计算 ====================
def align_bits_and_compute_ber(
    b_hat: np.ndarray,
    b_ref: np.ndarray,
    bits_per_sym: int,
    max_sym_shift: int = 2,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    在±max_sym_shift个symbol范围内搜索最优bit对齐
    
    Args:
        b_hat: 预测的比特序列
        b_ref: 参考的比特序列
        bits_per_sym: 每符号比特数
        max_sym_shift: 最大符号偏移数
        
    Returns:
        (最小BER, 对齐后的b_hat, 对齐后的b_ref, 使用的symbol shift)
    """
    assert bits_per_sym > 0
    best_ber = 1.0
    best_shift = 0
    best_pair = (None, None)

    Lh = len(b_hat)
    Lr = len(b_ref)

    for sym_shift in range(-max_sym_shift, max_sym_shift + 1):
        bit_shift = sym_shift * bits_per_sym

        if bit_shift >= 0:
            # b_hat向右移（丢前面）
            bh = b_hat[bit_shift:]
            br = b_ref[:len(bh)]
        else:
            # b_hat向左移（丢后面）
            bh = b_hat[:bit_shift]
            br = b_ref[-bit_shift:len(b_ref)]

        L = min(len(bh), len(br))
        if L <= 0:
            continue

        bh = bh[:L]
        br = br[:L]

        ber = np.mean(bh != br)

        if ber < best_ber:
            best_ber = ber
            best_shift = sym_shift
            best_pair = (bh.copy(), br.copy())

    if best_pair[0] is None:
        return 1.0, np.array([], dtype=np.int8), np.array([], dtype=np.int8), 0

    return best_ber, best_pair[0], best_pair[1], best_shift


def calculate_ser_from_symbols(pred_symbols: np.ndarray, 
                               true_symbols: np.ndarray,
                               modulation: str,
                               max_sym_shift: int = 8) -> float:
    """
    从符号序列计算SER（与test_sim_SignalSeparator.py一致）
    
    Args:
        pred_symbols: 预测的符号序列
        true_symbols: 真实的符号序列
        modulation: 调制方式
        max_sym_shift: 最大符号偏移数（用于对齐）
        
    Returns:
        SER值
    """
    if len(pred_symbols) == 0 or len(true_symbols) == 0:
        return 1.0
    
    modulation = modulation.upper()
    bits_per_sym = BITS_PER_SYMBOL.get(modulation, 2)
    
    # 解调
    pred_bits = demod_by_mod(pred_symbols, modulation)
    true_bits = demod_by_mod(true_symbols, modulation)
    
    # 比特对齐
    _, pred_bits_aligned, true_bits_aligned, _ = align_bits_and_compute_ber(
        pred_bits, true_bits, bits_per_sym=bits_per_sym, max_sym_shift=max_sym_shift
    )
    
    # 计算SER：按符号分组比较比特组合
    L_pred = len(pred_bits_aligned)
    L_true = len(true_bits_aligned)
    L_min = min(L_pred, L_true)
    L_min = (L_min // bits_per_sym) * bits_per_sym  # 向下取整到符号边界
    
    if L_min <= 0:
        return 1.0
    
    n_syms = L_min // bits_per_sym
    
    # 将比特序列重塑为(n_syms, bits_per_sym)形状
    pred_bits_syms = pred_bits_aligned[:L_min].reshape(n_syms, bits_per_sym)
    true_bits_syms = true_bits_aligned[:L_min].reshape(n_syms, bits_per_sym)
    
    # 比较每个符号的比特组合是否相同
    sym_errors = np.any(pred_bits_syms != true_bits_syms, axis=1)
    
    ser = float(np.mean(sym_errors)) if n_syms > 0 else 1.0
    return ser


# ==================== 实采数据解调函数 ====================
# 这些函数从 demodulate_baseband.py 和 split_from_raw_data.py 中提取
# 用于实采数据的解调（与仿真数据不同，需要Costas环补偿等）

def estimate_frequency_offset(signal_complex: np.ndarray, fs: float, sps: int, modulation: str = 'QPSK') -> float:
    """
    使用FFT方法估计频偏（从demodulate_baseband.py提取）
    
    Args:
        signal_complex: 输入复数信号
        fs: 采样频率
        sps: 每符号采样数
        modulation: 调制方式
        
    Returns:
        频偏（Hz）
    """
    from scipy.fft import fft, fftfreq
    
    modulation = modulation.upper()
    
    # 根据调制方式选择去调制的幂次
    if modulation == 'QPSK':
        power_order = 4
    elif modulation == '8PSK':
        power_order = 8
    elif modulation == '16QAM':
        power_order = 4  # 近似
    else:
        power_order = 4
    
    # 对信号进行N次方运算以去除调制
    signal_power_n = signal_complex ** power_order
    
    # 计算FFT
    nfft = len(signal_power_n)
    fft_result = fft(signal_power_n, nfft)
    freqs = fftfreq(nfft, 1/fs)
    
    # 找到峰值频率
    power_spectrum = np.abs(fft_result)
    positive_freq_idx = freqs >= 0
    positive_freqs = freqs[positive_freq_idx]
    positive_power = power_spectrum[positive_freq_idx]
    
    # 找到峰值
    peak_idx = np.argmax(positive_power)
    peak_freq = positive_freqs[peak_idx]
    
    # 频偏是峰值的1/N
    freq_offset = peak_freq / float(power_order)
    
    return freq_offset


def compensate_cfo_phase(signal_complex: np.ndarray, freq_offset_hz: float, initial_phase: float, fs: float) -> np.ndarray:
    """
    补偿频偏和相偏（从demodulate_baseband.py提取）
    
    Args:
        signal_complex: 输入信号
        freq_offset_hz: 频偏（Hz）
        initial_phase: 初始相位（rad）
        fs: 采样频率
        
    Returns:
        补偿后的信号
    """
    n = np.arange(len(signal_complex))
    t = n / fs
    correction = np.exp(-1j * (2 * np.pi * freq_offset_hz * t + initial_phase))
    return signal_complex * correction


def costas_loop_enhanced(signal_complex: np.ndarray, loop_bandwidth: float, sps: int, fs: float, modulation: str = 'QPSK') -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Costas环补偿（从demodulate_baseband.py提取）
    
    Args:
        signal_complex: 输入复数信号
        loop_bandwidth: 环路带宽
        sps: 每符号采样数
        fs: 采样频率
        modulation: 调制方式
        
    Returns:
        (补偿后的信号, 估计的频偏, 初始相位, 相位历史)
    """
    # 先估计频偏
    freq_offset_est = estimate_frequency_offset(signal_complex, fs, sps, modulation)
    
    # 粗略补偿频偏
    signal_rough = compensate_cfo_phase(signal_complex, freq_offset_est, 0, fs)
    
    # 使用Costas环精细调整相位
    phase_estimate = 0
    phase_history = []
    corrected_signal = []
    
    # 环路滤波器参数
    alpha = loop_bandwidth  # 比例项
    beta = alpha * alpha   # 积分项
    
    integrator = 0
    
    for i in range(len(signal_rough)):
        # 相位旋转补偿
        current_sample = signal_rough[i] * np.exp(-1j * phase_estimate)
        corrected_signal.append(current_sample)
        
        # 相位误差检测
        real_part = np.real(current_sample)
        imag_part = np.imag(current_sample)
        
        # 计算误差
        error = np.sign(real_part) * imag_part - np.sign(imag_part) * real_part
        
        # 环路滤波
        integrator = integrator + beta * error
        phase_step = alpha * error + integrator
        
        # 更新相位估计
        phase_estimate += phase_step
        phase_history.append(phase_estimate)
    
    # 从相位历史估计初始相位
    initial_phase = phase_history[0] if len(phase_history) > 0 else 0
    
    return np.array(corrected_signal), freq_offset_est, initial_phase, np.array(phase_history)


def evaluate_clustering_quality(symbols: np.ndarray, modulation: str = 'QPSK') -> float:
    """
    评估星座图的聚类质量（从demodulate_baseband.py提取）
    
    Args:
        symbols: 复数符号序列
        modulation: 调制方式
        
    Returns:
        聚类质量得分（越高越好）
    """
    if len(symbols) == 0:
        return -np.inf
    
    modulation = modulation.upper()
    
    # 定义理想星座点
    if modulation == 'QPSK':
        ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif modulation == '8PSK':
        angles = np.arange(8) * 2 * np.pi / 8
        ideal_points = np.exp(1j * angles)
    elif modulation == '16QAM':
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
        ideal_points = []
        for I in levels:
            for Q in levels:
                ideal_points.append(I + 1j * Q)
        ideal_points = np.array(ideal_points)
    else:
        ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    # 归一化输入符号的能量
    power = np.mean(np.abs(symbols)**2)
    if power == 0:
        return -np.inf
    normalized_symbols = symbols / np.sqrt(power)
    
    # 归一化理想点
    ideal_power = np.mean(np.abs(ideal_points)**2)
    if ideal_power > 0:
        ideal_points = ideal_points / np.sqrt(ideal_power)
    
    # 计算每个符号到所有理想点的距离
    distances = np.abs(normalized_symbols[:, np.newaxis] - ideal_points)
    
    # 找到每个符号最近的理想点及其距离
    min_distances = np.min(distances, axis=1)
    
    # 计算平均最小距离
    mean_error_distance = np.mean(min_distances)
    
    # 返回平均距离的倒数作为得分
    return 1.0 / (mean_error_distance + 1e-9)


def find_optimal_loop_bandwidth(signal_slice: np.ndarray, sps: int, fs: float, modulation: str,
                                target_score: float = 6.0, min_bw: float = 1e-14, max_bw: float = 1e-5,
                                num_trials: int = 50, verbose: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray, int]:
    """
    搜索最优的loop_bandwidth（从demodulate_baseband.py提取，简化版）
    
    Args:
        signal_slice: 输入信号切片
        sps: 每符号采样数
        fs: 采样频率
        modulation: 调制方式
        target_score: 目标聚类质量分数
        min_bw: 最小loop_bandwidth
        max_bw: 最大loop_bandwidth
        num_trials: 搜索尝试次数
        verbose: 是否打印详细信息
        
    Returns:
        (best_bw, best_score, best_phase_history, best_signal, best_offset)
    """
    
    best_bw = min_bw
    best_score = -np.inf
    best_phase_history = None
    best_signal = None
    best_offset = -1
    
    # 生成测试值（对数空间均匀分布）
    log_min = np.log10(min_bw)
    log_max = np.log10(max_bw)
    test_values = np.logspace(log_min, log_max, num_trials)
    
    for test_bw in test_values:
        try:
            # 使用Costas环补偿
            signal_compensated, _, _, phase_history = costas_loop_enhanced(
                signal_slice, loop_bandwidth=test_bw, sps=sps, fs=fs, modulation=modulation
            )
            
            # 匹配滤波
            rrc_filter_coeff = rrc_filter(BETA, sps, NUM_TAPS)
            signal_real = signal_module.lfilter(rrc_filter_coeff, 1, signal_compensated.real)
            signal_imag = signal_module.lfilter(rrc_filter_coeff, 1, signal_compensated.imag)
            signal_mf = signal_real + 1j * signal_imag
            
            # 补偿滤波器延迟
            delay = NUM_TAPS // 2
            signal_mf = signal_mf[delay:]
            
            # 评估不同offset的聚类质量
            best_offset_temp = -1
            best_score_temp = -np.inf
            
            for offset in range(sps):
                symbols_temp = signal_mf[offset::sps]
                if len(symbols_temp) > 0:
                    current_score = evaluate_clustering_quality(symbols_temp, modulation)
                    if current_score > best_score_temp:
                        best_score_temp = current_score
                        best_offset_temp = offset
            
            # 如果达到目标分数，记录并继续寻找更小的值
            if best_score_temp >= target_score:
                if best_score < target_score or test_bw < best_bw:
                    best_bw = test_bw
                    best_score = best_score_temp
                    best_phase_history = phase_history
                    best_signal = signal_mf  # 保存匹配滤波后的信号
                    best_offset = best_offset_temp
            # 如果还没达到目标，但比当前最好结果好，也记录
            elif best_score_temp > best_score:
                best_bw = test_bw
                best_score = best_score_temp
                best_phase_history = phase_history
                best_signal = signal_mf
                best_offset = best_offset_temp
                
        except Exception as e:
            if verbose:
                print(f"    loop_bandwidth={test_bw:.2e}: 失败 - {e}")
            continue
    
    return best_bw, best_score, best_phase_history, best_signal, best_offset


def demodulate_real_signal(signal: np.ndarray, modulation: str,
                          loop_bandwidth: Optional[float] = None,
                          auto_search: bool = True,
                          threshold: float = 6.0) -> Tuple[np.ndarray, float, Dict]:
    """
    解调实采数据信号（与split_from_raw_data.py的evaluate_single_slice_standalone一致）
    
    Args:
        signal: 输入复数信号
        modulation: 调制方式
        loop_bandwidth: Costas环带宽（如果为None且auto_search=True，则自动搜索）
        auto_search: 是否自动搜索最优loop_bandwidth
        threshold: 聚类质量阈值（用于自动搜索时的target_score）
        
    Returns:
        (symbols, score, info_dict)
        - symbols: 解调得到的符号
        - score: 聚类质量得分
        - info_dict: 包含解调信息的字典
    """
    
    signal = np.asarray(signal, dtype=np.complex128)
    modulation = modulation.upper()
    
    # 如果auto_search且未指定loop_bandwidth，自动搜索
    if loop_bandwidth is None and auto_search:
        optimal_bw, score_temp, _, signal_mf, offset_pre_mf = find_optimal_loop_bandwidth(
            signal, sps=SPS, fs=FS, modulation=modulation,
            target_score=threshold,
            min_bw=1e-14,
            max_bw=1e-5,
            num_trials=50,
            verbose=False
        )
        
        # 匹配滤波后的信号已经处理好了，直接抽取符号
        symbols = signal_mf[offset_pre_mf::SPS]
        
        # 最终评估聚类质量
        final_score = evaluate_clustering_quality(symbols, modulation)
        
        info_dict = {
            'loop_bandwidth': optimal_bw,
            'offset': offset_pre_mf,
            'freq_offset_est': None,
            'initial_phase': None,
            'num_symbols': len(symbols),
            'clustering_score': final_score
        }
        
        return symbols, final_score, info_dict
    else:
        # 使用指定的loop_bandwidth或默认值
        if loop_bandwidth is None:
            loop_bandwidth = 1e-7  # 默认值
        
        # Costas环补偿
        signal_compensated, freq_offset_est, initial_phase, phase_history = costas_loop_enhanced(
            signal, loop_bandwidth=loop_bandwidth, sps=SPS, fs=FS, modulation=modulation
        )
        
        
        rrc_filter_coeff = rrc_filter(BETA, SPS, NUM_TAPS)
        signal_real = signal_module.lfilter(rrc_filter_coeff, 1, signal_compensated.real)
        signal_imag = signal_module.lfilter(rrc_filter_coeff, 1, signal_compensated.imag)
        signal_mf = signal_real + 1j * signal_imag
        
        # 补偿滤波器延迟
        delay = NUM_TAPS // 2
        signal_mf = signal_mf[delay:]
        
        # 评估不同offset的聚类质量，找最佳offset
        best_offset = 0
        best_score = -np.inf
        
        for offset in range(SPS):
            symbols_temp = signal_mf[offset::SPS]
            if len(symbols_temp) > 0:
                score = evaluate_clustering_quality(symbols_temp, modulation)
                if score > best_score:
                    best_score = score
                    best_offset = offset
        
        # 使用最佳offset抽取符号
        symbols = signal_mf[best_offset::SPS]
        
        # 最终评估聚类质量
        final_score = evaluate_clustering_quality(symbols, modulation)
        
        info_dict = {
            'loop_bandwidth': loop_bandwidth,
            'offset': best_offset,
            'freq_offset_est': freq_offset_est,
            'initial_phase': initial_phase,
            'num_symbols': len(symbols),
            'clustering_score': final_score
        }
        
        return symbols, final_score, info_dict

# ==================== 下采样函数 ====================
def downsample_to_sps8(signal_complex, original_sps):
    """
    将信号从original_sps下采样到sps=8
    """
    if original_sps == TARGET_SPS:
        return signal_complex
    
    # 计算下采样因子
    downsample_factor = original_sps // TARGET_SPS
    
    # 使用抗混叠滤波器
    if downsample_factor > 1:
        # 设计低通滤波器
        nyquist = TARGET_SPS / 2
        cutoff = nyquist * 0.8  # 留一些余量
        b, a = signal_module.butter(4, cutoff / (original_sps / 2), 'low')
        signal_filtered = signal_module.filtfilt(b, a, signal_complex)
        
        # 下采样
        signal_downsampled = signal_filtered[::downsample_factor]
    else:
        signal_downsampled = signal_complex
    
    return signal_downsampled


# ==================== 硬判决函数 ====================
def qpsk_hard_decision(symbols: np.ndarray) -> np.ndarray:
    """
    QPSK硬判决：将符号映射到最近的理想星座点
    
    QPSK星座点（归一化）：
    - 00: (+1+1j)/sqrt(2)
    - 01: (-1+1j)/sqrt(2)
    - 10: (+1-1j)/sqrt(2)
    - 11: (-1-1j)/sqrt(2)
    """
    ideal_points = np.array([
        (1 + 1j) / np.sqrt(2),   # 00
        (-1 + 1j) / np.sqrt(2),  # 01
        (1 - 1j) / np.sqrt(2),   # 10
        (-1 - 1j) / np.sqrt(2),  # 11
    ])
    
    result = np.zeros_like(symbols, dtype=np.complex128)
    for i, sym in enumerate(symbols):
        # 找最近的星座点
        distances = np.abs(sym - ideal_points)
        idx = np.argmin(distances)
        result[i] = ideal_points[idx]
    
    return result


def psk8_hard_decision(symbols: np.ndarray) -> np.ndarray:
    """
    8PSK硬判决：将符号映射到最近的理想星座点
    
    8PSK星座点：exp(j * 2πk/8), k=0,1,...,7
    """
    ideal_points = np.array([np.exp(1j * 2 * np.pi * k / 8.0) for k in range(8)])
    
    result = np.zeros_like(symbols, dtype=np.complex128)
    for i, sym in enumerate(symbols):
        # 找最近的星座点
        distances = np.abs(sym - ideal_points)
        idx = np.argmin(distances)
        result[i] = ideal_points[idx]
    
    return result


def qam16_hard_decision(symbols: np.ndarray) -> np.ndarray:
    """
    16QAM硬判决：将符号映射到最近的理想星座点
    
    16QAM星座点（Gray编码）：
    - I, Q ∈ {-3, -1, 1, 3} / sqrt(10)
    """
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10.0)
    
    result = np.zeros_like(symbols, dtype=np.complex128)
    for i, sym in enumerate(symbols):
        # I和Q分量分别找最近的level
        I = sym.real
        Q = sym.imag
        
        idx_I = np.argmin(np.abs(I - levels))
        idx_Q = np.argmin(np.abs(Q - levels))
        
        result[i] = levels[idx_I] + 1j * levels[idx_Q]
    
    return result


# ==================== SNR估计与噪声添加 ====================
def estimate_signal_snr(signal: np.ndarray, modulation: str = "QPSK", 
                       sps: int = 8, filter_type: str = "RRC") -> tuple:
    """
    估计信号的信噪比（使用脉冲响应估计的重构方法）
    
    方法（参考estimate_h.py）：
    1. 使用Costas环补偿频偏和相偏
    2. 找到最佳offset，抽取符号
    3. 硬判决得到理想符号
    4. 生成理想方波信号
    5. 使用最小二乘法估计脉冲响应
    6. 用估计的脉冲响应重构信号
    7. 计算噪声和SNR
    
    参数：
        signal: 复数基带信号
        modulation: 调制方式 (QPSK, 8PSK, 16QAM)
        sps: 每符号采样数
        filter_type: 滤波器类型 (RRC 或 RC) - 仅用于硬判决，不影响重构
    
    返回：
        (snr_db, clean_signal, noise_signal): 信噪比(dB), 重构的干净信号, 估计的噪声信号
    
    注意：
        - clean_signal是补偿后的重构信号
        - noise_signal是相对于补偿后信号的噪声
    """
    signal = np.asarray(signal, dtype=np.complex128)
    modulation = modulation.upper()
    
    try:
        # 1. 解调信号获取最佳offset和聚类分数
        symbols, score, info_dict = demodulate_real_signal(
            signal, modulation, auto_search=True, threshold=6.0
        )
        
        if len(symbols) == 0:
            # 解调失败，返回默认值
            return 999.0, signal, np.zeros_like(signal)
        
        best_offset = info_dict.get('offset', 0) or 0
        loop_bandwidth = info_dict.get('loop_bandwidth', 1e-7) or 1e-7
        
    except Exception as e:
        # 解调失败，返回默认值
        return 999.0, signal, np.zeros_like(signal)
    
    try:
        # 2. 使用Costas环补偿频偏和相偏
        signal_compensated, freq_offset_est, initial_phase, phase_history = costas_loop_enhanced(
            signal, loop_bandwidth=loop_bandwidth, sps=sps, fs=FS, modulation=modulation
        )
        
        # 3. 从补偿后的信号中抽取符号（使用最佳offset）
        symbols_rx = signal_compensated[best_offset::sps]
        
        # 4. 硬判决得到理想符号
        if modulation == "QPSK":
            ideal_symbols = qpsk_hard_decision(symbols_rx)
        elif modulation == "8PSK":
            ideal_symbols = psk8_hard_decision(symbols_rx)
        elif modulation == "16QAM":
            ideal_symbols = qam16_hard_decision(symbols_rx)
        else:
            raise ValueError(f"不支持的调制方式: {modulation}")
        
        # 5. 生成理想方波信号（在对应采样点放置判决符号）
        ideal_signal = np.zeros_like(signal_compensated, dtype=complex)
        ideal_signal[best_offset::sps] = ideal_symbols
        
        # 6. 使用最小二乘法估计脉冲响应
        N_filter = 8 * sps  # 滤波器长度
        
        # 构建矩阵A（理想信号）
        A_rows = len(signal_compensated) - N_filter + 1
        if A_rows <= 0:
            # 信号太短，无法估计脉冲响应
            return 999.0, signal_compensated, np.zeros_like(signal_compensated)
        
        A = np.zeros((A_rows, N_filter), dtype=complex)
        for n in range(A_rows):
            A[n, :] = ideal_signal[n:n + N_filter]
        
        # 构建AA矩阵（实部和虚部堆叠）
        AA = np.vstack([np.real(A), np.imag(A)])
        
        # 提取对应的实部和虚部数据（补偿后的信号）
        xxI = np.real(signal_compensated)[N_filter//2 : N_filter//2 + A_rows]
        xxQ = np.imag(signal_compensated)[N_filter//2 : N_filter//2 + A_rows]
        
        # 构建目标向量
        xxxx = np.hstack([xxI, xxQ])
        
        # 求解最小二乘问题
        h = np.linalg.lstsq(AA, xxxx, rcond=None)[0]
        h_estimated = np.flipud(h)  # 翻转
        
        # 7. 找到最优延迟并应用
        def find_optimal_delay(reference, measured, max_delay=10):
            """找到最优延迟量"""
            correlations = []
            for delay in range(-max_delay, max_delay+1):
                if delay >= 0:
                    ref_shifted = reference[delay:]
                    meas_trimmed = measured[:len(ref_shifted)]
                else:
                    ref_shifted = reference[:delay] if delay != 0 else reference
                    meas_trimmed = measured[-delay:len(ref_shifted)-delay] if delay != 0 else measured
                    
                if len(ref_shifted) > 0 and len(ref_shifted) == len(meas_trimmed):
                    correlation = np.corrcoef(ref_shifted, meas_trimmed)[0, 1]
                    correlations.append((delay, correlation))
            
            if correlations:
                optimal_delay, max_corr = max(correlations, key=lambda x: abs(x[1]))
                return optimal_delay
            return 0
        
        def apply_delay_signal(sig, delay):
            """应用延迟到信号"""
            if delay > 0:
                return np.concatenate([np.zeros(delay, dtype=sig.dtype), sig[:-delay]])
            elif delay < 0:
                return sig[-delay:]
            else:
                return sig
        
        optimal_delay = find_optimal_delay(ideal_signal, signal_compensated)
        ideal_signal_aligned = apply_delay_signal(ideal_signal, optimal_delay)
        
        # 8. 用估计的脉冲响应重构信号
        reconstructed_signal = signal_module.convolve(ideal_signal_aligned, h_estimated, mode='same')
        
        # 9. 计算噪声
        noise = reconstructed_signal - signal_compensated
        
        # 10. 截去前25个采样点（前面几个点估计不准）
        skip_samples = 25
        if len(reconstructed_signal) > skip_samples:
            reconstructed_signal_trimmed = reconstructed_signal[skip_samples:]
            signal_compensated_trimmed = signal_compensated[skip_samples:]
            noise_trimmed = noise[skip_samples:]
        else:
            reconstructed_signal_trimmed = reconstructed_signal
            signal_compensated_trimmed = signal_compensated
            noise_trimmed = noise
        
        # 11. 计算功率和SNR（使用截取后的信号）
        # SNR = 重构信号功率 / 噪声功率
        signal_power = np.mean(np.abs(reconstructed_signal_trimmed) ** 2)
        noise_power = np.mean(np.abs(noise_trimmed) ** 2)
        
        if noise_power < 1e-12:
            snr_db = 999.0
        else:
            snr_db = 10 * np.log10(signal_power / noise_power)
        
        # 返回时保持原始长度，但SNR是基于截取后的信号计算的
        return snr_db, reconstructed_signal, noise
        
    except Exception as e:
        # 重构失败，返回默认值
        return 999.0, signal, np.zeros_like(signal)


def add_noise_to_target_snr(signal: np.ndarray, current_snr_db: float, 
                            target_snr_db: float, seed: int = None) -> np.ndarray:
    """
    添加AWGN噪声，使信号达到目标信噪比
    
    参数：
        signal: 输入信号
        current_snr_db: 当前信噪比(dB)
        target_snr_db: 目标信噪比(dB)
        seed: 随机种子（用于可重复性）
    
    返回：
        noisy_signal: 加噪后的信号
    """
    if current_snr_db <= target_snr_db:
        # 当前SNR已经低于或等于目标SNR，不需要加噪
        return signal
    
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 计算目标噪声功率
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    
    # 计算当前噪声功率
    current_noise_power = signal_power / (10 ** (current_snr_db / 10))
    
    # 计算需要添加的噪声功率
    added_noise_power = target_noise_power - current_noise_power
    
    if added_noise_power <= 0:
        return signal
    
    # 生成AWGN噪声
    if seed is not None:
        np.random.seed(seed)
    
    noise_std = np.sqrt(added_noise_power / 2)  # 复数噪声，实部和虚部各一半功率
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    # 添加噪声
    noisy_signal = signal + noise
    
    return noisy_signal


def estimate_and_add_noise_if_needed(signal: np.ndarray, modulation: str = "QPSK",
                                     target_snr_db: float = 15.0, sps: int = 8,
                                     filter_type: str = "RRC", seed: int = None) -> tuple:
    """
    估计信号SNR，如果低于目标SNR则添加噪声
    
    参数：
        signal: 输入信号
        modulation: 调制方式
        target_snr_db: 目标信噪比(dB)
        sps: 每符号采样数
        filter_type: 滤波器类型
        seed: 随机种子
    
    返回：
        (processed_signal, actual_snr_db, noise_added): 处理后的信号, 实际SNR(dB), 是否添加了噪声
    """
    # 估计当前SNR
    current_snr_db, clean_signal, noise_signal = estimate_signal_snr(
        signal, modulation, sps, filter_type
    )
    
    # 判断是否需要添加噪声
    if current_snr_db > target_snr_db:
        # 当前SNR高于目标，添加噪声降低到目标SNR
        processed_signal = add_noise_to_target_snr(
            signal, current_snr_db, target_snr_db, seed
        )
        actual_snr_db = target_snr_db
        noise_added = True
    else:
        # 当前SNR已经低于目标，不添加噪声
        processed_signal = signal
        actual_snr_db = current_snr_db
        noise_added = False
    
    return processed_signal, actual_snr_db, noise_added
