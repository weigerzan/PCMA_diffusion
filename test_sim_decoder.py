# test_sim_SignalSeparator.py
# -*- coding: utf-8 -*-
import os, re, math, argparse, numpy as np, pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy.signal import convolve

# from compensation import costas_loop  # 目前没用到，但保留
# from model_complex import SignalSeparator

from datasets.decoding_generate_dataset import PCMAModDataset_Decoding_Generate
# import argparse
from utils.config import load_config



torch.backends.cudnn.benchmark = True
# import numpy as np

def batch_snr_db(p, g, eps=1e-12):
    """
    计算逐样本 LS 幅度对齐后的重建 SNR（dB）

    Args:
        p: np.ndarray, shape (B, 2, T), 预测信号 (IQ)
        g: np.ndarray, shape (B, 2, T), 真值信号 (IQ)
        eps: 数值稳定项

    Returns:
        snr_db_mean: float, batch 平均 SNR (dB)
        snr_db_each: np.ndarray, shape (B,), 每个样本的 SNR (dB)
    """
    assert p.shape == g.shape
    assert p.ndim == 3 and p.shape[1] == 2

    # IQ -> complex: (B, T)
    p_c = p[:, 0, :] + 1j * p[:, 1, :]
    g_c = g[:, 0, :] + 1j * g[:, 1, :]

    # -------- LS 幅度对齐 --------
    # alpha = <p, g> / <p, p>
    # vdot: conjugate(p) * g 再求和
    num = np.sum(np.conj(p_c) * g_c, axis=1)          # (B,)
    den = np.sum(np.abs(p_c) ** 2, axis=1) + eps      # (B,)
    alpha = num / den                                 # (B,)

    # 对齐后的预测
    p_aligned = alpha[:, None] * p_c                  # (B, T)

    # -------- SNR --------
    signal_power = np.mean(np.abs(g_c) ** 2, axis=1)  # (B,)
    noise_power  = np.mean(np.abs(g_c - p_aligned) ** 2, axis=1) + eps

    snr_db_each = 10.0 * np.log10(signal_power / noise_power)
    snr_db_mean = float(np.mean(snr_db_each))

    # return snr_db_mean, snr_db_each
    return snr_db_each

def align_bits_and_compute_ber(
    b_hat: np.ndarray,
    b_ref: np.ndarray,
    bits_per_sym: int,
    max_sym_shift: int = 2,
):
    """
    在 ±max_sym_shift 个 symbol 范围内搜索最优 bit 对齐，
    返回：
      - 最小 BER
      - 对齐后的 b_hat
      - 对齐后的 b_ref
      - 使用的 symbol shift（整数）
    """

    assert bits_per_sym > 0
    best_ber = 1.0
    best_shift = 0
    best_pair = (None, None)

    Lh = len(b_hat)
    Lr = len(b_ref)
    # print(bits_per_sym)
    # print(max_sym_shift)
    for sym_shift in range(-max_sym_shift, max_sym_shift + 1):
        # for k in range(bits_per_sym):
        # print(sym_shift)
        bit_shift = sym_shift * bits_per_sym

        if bit_shift >= 0:
            # b_hat 向右移（丢前面）
            bh = b_hat[bit_shift:]
            br = b_ref[:len(bh)]
        else:
            # b_hat 向左移（丢后面）
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
    # print(best_shift)
    # print(best_ber)
    if best_pair[0] is None:
        return 1.0, np.array([]), np.array([]), 0

    return best_ber, best_pair[0], best_pair[1], best_shift

# ==================== 分布式工具 ====================
def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if dist_is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist_is_initialized() else 1

def setup_ddp(backend="nccl"):
    if dist_is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")

def cleanup_ddp():
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ==================== DSP 基本件 ====================
beta, sps, num_taps = 0.33, 8, 64
fs = 12e6

def rc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))
    return h

def rrc_filter(beta, sps, num_taps):
    """
    RRC（Root-Raised-Cosine）滤波器，符号间隔 Ts=1。
    使用标准闭式形式，并处理 t=0 和 t=±Ts/(4β) 的极限值。
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

rc = rc_filter(beta, sps, num_taps)
rrc = rrc_filter(beta, sps, num_taps)


# ==================== 调制相关：bit/符号数 ====================
BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

# ==================== 解调函数 ====================
def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """
    与 qpsk_mod 对应：
      00 -> (+,+)
      01 -> (-,+)
      10 -> (+,-)
      11 -> (-,-)
    """
    bits = []
    sym = symbols * np.sqrt(2)
    for s in sym:
        if s.real >= 0 and s.imag >= 0: b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0: b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0: b1, b2 = 1, 0
        else: b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits, dtype=np.int8)

def psk8_demod(symbols: np.ndarray) -> np.ndarray:
    """
    对应 8PSK：
      bits -> k = b0*4 + b1*2 + b2 (见你之前的调制实现)
      s = exp(j * 2πk/8)
    这里反过来：从角度恢复 k，再还原 bits (b0,b1,b2)。
    """
    angles = np.angle(symbols)
    angles = np.mod(angles, 2*np.pi)
    step = 2*np.pi / 8.0
    # k = np.round(angles / step).astype(int) % 8  # 0..7
    k = np.floor((angles + step/2) / step).astype(int) % 8
    bits = []
    for val in k:
        b0 = (val >> 2) & 1  # 权重 4
        b1 = (val >> 1) & 1  # 权重 2
        b2 = val & 1         # 权重 1
        bits.extend([b0, b1, b2])
    return np.array(bits, dtype=np.int8)

def qam16_demod(symbols: np.ndarray) -> np.ndarray:
    """
    对应 16QAM：
      I,Q ∈ {-3,-1,1,3}/sqrt(10)，Gray 编码：
        -3 -> 00, -1 -> 01, 1 -> 11, 3 -> 10
    这里按最近邻找 I/Q 所在的 level，再映射回 bits。
    """
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10.0)
    bits = []
    for s in symbols:
        I = s.real
        Q = s.imag
        idx_I = np.argmin((I - levels)**2)
        idx_Q = np.argmin((Q - levels)**2)
        level_I = levels[idx_I]
        level_Q = levels[idx_Q]

        # level -> Gray bits
        if level_I < (-2/np.sqrt(10)):   bi0, bi1 = 0, 0   # -3
        elif level_I < 0:                bi0, bi1 = 0, 1   # -1
        elif level_I > (2/np.sqrt(10)):  bi0, bi1 = 1, 0   # 3
        else:                            bi0, bi1 = 1, 1   # 1

        if level_Q < (-2/np.sqrt(10)):   bq0, bq1 = 0, 0
        elif level_Q < 0:                bq0, bq1 = 0, 1
        elif level_Q > (2/np.sqrt(10)):  bq0, bq1 = 1, 0
        else:                            bq0, bq1 = 1, 1

        bits.extend([bi0, bi1, bq0, bq1])
    return np.array(bits, dtype=np.int8)

def demod_by_mod(symbols: np.ndarray, modulation: str) -> np.ndarray:
    modulation = (modulation or "QPSK").upper()
    if modulation == "QPSK":
        return qpsk_demod(symbols)
    elif modulation == "8PSK":
        return psk8_demod(symbols)
    elif modulation == "16QAM":
        return qam16_demod(symbols)
    else:
        # 默认当 QPSK 处理，避免崩溃
        return qpsk_demod(symbols)

# ==================== 其它工具 ====================
def align_phase(ref, est):
    c = np.mean(ref * np.conj(est) + 1e-12)
    a = np.angle(c)
    return est * np.exp(-1j * a)

def wrap_2pi(x): return np.mod(x, 2*np.pi)

def evm_rms(ref_syms, est_syms):
    num = np.mean(np.abs(est_syms - ref_syms)**2)
    den = np.mean(np.abs(ref_syms)**2) + 1e-12
    return float(np.sqrt(num / den))

def find_best_offset(y_mf, sps):
    best_off = 0
    best_eng = -1.0
    for off in range(sps):
        sym = y_mf[off::sps]
        eng = np.mean(np.abs(sym)**2)
        if eng > best_eng:
            best_eng = eng
            best_off = off
    return best_off

def mf_and_sample(wave, sps, rc, num_taps, guard_sym=None):
    if guard_sym is None:
        guard_sym = num_taps // sps  # 64/8=8 符号

    if wave is None or len(wave) == 0:
        return np.zeros(0, dtype=np.complex64)

    # y_mf = convolve(wave, rrc, mode='same')
    y_mf = wave
    off = find_best_offset(y_mf, sps)
    # print(off)
    syms = y_mf[off::sps]
    if len(syms) <= 2 * guard_sym:
        return np.zeros(0, dtype=np.complex64)
    syms = syms[guard_sym:-guard_sym]

    m = np.mean(np.abs(syms))
    if m > 0:
        syms = syms / m
    return syms.astype(np.complex64)

def slice_bits_to_match_syms(bits_full: np.ndarray, n_syms_used: int, bits_per_sym: int):
    """
    通用版：支持 QPSK/8PSK/16QAM
    """
    if len(bits_full) == 0 or n_syms_used <= 0 or bits_per_sym <= 0:
        return np.zeros(0, dtype=np.int8)

    n_sym_total = len(bits_full) // bits_per_sym
    n_syms_used = min(n_syms_used, n_sym_total)
    if n_sym_total <= n_syms_used:
        return bits_full[:bits_per_sym * n_syms_used]

    guard_sym = max((n_sym_total - n_syms_used) // 2, 0)
    start = guard_sym * bits_per_sym
    end = start + bits_per_sym * n_syms_used
    end = min(end, len(bits_full))
    return bits_full[start:end]

# ==================== 解析 params ====================
def parse_params_all(pstr: str):
    """
    通用解析：兼容旧的 QPSK test_all 和新的 test_snr_amp/test_cfo_phase/test_delay
    """
    if not isinstance(pstr, str) or not pstr:
        return {}
    parts = [x.strip() for x in pstr.split(',')]
    floats = []
    for x in parts:
        try: floats.append(float(x))
        except: pass
    snr = floats[0] if len(floats) >= 1 else None
    amp = floats[1] if len(floats) >= 2 else None

    m_f1   = re.search(r"f_off1=([\-0-9.]+)\s*Hz", pstr)
    m_f2   = re.search(r"f_off2=([\-0-9.]+)\s*Hz", pstr)
    m_p1   = re.search(r"phi1=([\-0-9.]+)\s*rad", pstr)
    m_p2   = re.search(r"phi2=([\-0-9.]+)\s*rad", pstr)
    m_rep  = re.search(r"rep=([0-9]+)", pstr)
    m_d1   = re.search(r"delay1_samp=([\-0-9]+)", pstr)
    m_d2   = re.search(r"delay2_samp=([\-0-9]+)", pstr)
    m_dd   = re.search(r"delay_diff_samp=([\-0-9]+)", pstr)
    m_mod1 = re.search(r"mod1=([A-Za-z0-9]+)", pstr)
    m_mod2 = re.search(r"mod2=([A-Za-z0-9]+)", pstr)

    mod1 = m_mod1.group(1) if m_mod1 else None
    mod2 = m_mod2.group(1) if m_mod2 else None
    # 兼容旧的 QPSK test_all：params 末尾有 "QPSK"
    if (mod1 is None or mod2 is None) and ("QPSK" in pstr):
        mod1 = mod1 or "QPSK"
        mod2 = mod2 or "QPSK"

    return {
        'snr': snr,
        'amp': amp,
        'f_off1': float(m_f1.group(1)) if m_f1 else None,
        'f_off2': float(m_f2.group(1)) if m_f2 else None,
        'phi1': float(m_p1.group(1)) if m_p1 else None,
        'phi2': float(m_p2.group(1)) if m_p2 else None,
        'rep': int(m_rep.group(1)) if m_rep else None,
        'delay1': int(m_d1.group(1)) if m_d1 else None,
        'delay2': int(m_d2.group(1)) if m_d2 else None,
        'delay_diff': int(m_dd.group(1)) if m_dd else None,
        'mod1': mod1,
        'mod2': mod2,
    }

# ==================== Dataset ====================
class GenericTestDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        def c2ri(x):
            x = np.asarray(x)
            return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=0)

        raw = e.get('params', "")
        params_str = ", ".join(map(str, raw)) if isinstance(raw, (list, tuple)) else str(raw)
        b1 = np.asarray(e.get('bits1', np.array([-1], dtype=np.int8)), dtype=np.int8)
        b2 = np.asarray(e.get('bits2', np.array([-1], dtype=np.int8)), dtype=np.int8)
        return {
            'mixsignal_ri': c2ri(e['mixsignal']),
            'rfsignal1_ri': c2ri(e['rfsignal1']),
            'rfsignal2_ri': c2ri(e['rfsignal2']),
            'bits1': b1,
            'bits2': b2,
            'params': params_str,
        }

# ==================== Collate Function ====================
def collate_fn_pad(batch):
    """
    自定义 collate 函数，处理长度不一致的信号。
    将所有信号 padding 到 batch 内的最大长度。
    """
    # 找到 batch 内所有信号的最大长度
    max_len = 0
    for sample in batch:
        max_len = max(max_len, sample['mixsignal_ri'].shape[1])
        max_len = max(max_len, sample['rfsignal1_ri'].shape[1])
        max_len = max(max_len, sample['rfsignal2_ri'].shape[1])
    
    # 收集所有需要 padding 的 tensor
    mixsignal_list = []
    rfsignal1_list = []
    rfsignal2_list = []
    bits1_list = []
    bits2_list = []
    params_list = []
    original_lengths = []
    
    for sample in batch:
        mix_ri = sample['mixsignal_ri']  # (2, T)
        rf1_ri = sample['rfsignal1_ri']  # (2, T)
        rf2_ri = sample['rfsignal2_ri']  # (2, T)
        
        # 记录原始长度
        orig_len = mix_ri.shape[1]
        original_lengths.append(orig_len)
        
        # Padding 到最大长度（在最后一个维度）
        if mix_ri.shape[1] < max_len:
            pad_width = max_len - mix_ri.shape[1]
            mix_ri = np.pad(mix_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        if rf1_ri.shape[1] < max_len:
            pad_width = max_len - rf1_ri.shape[1]
            rf1_ri = np.pad(rf1_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        if rf2_ri.shape[1] < max_len:
            pad_width = max_len - rf2_ri.shape[1]
            rf2_ri = np.pad(rf2_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        mixsignal_list.append(torch.from_numpy(mix_ri))
        rfsignal1_list.append(torch.from_numpy(rf1_ri))
        rfsignal2_list.append(torch.from_numpy(rf2_ri))
        bits1_list.append(torch.from_numpy(sample['bits1']))
        bits2_list.append(torch.from_numpy(sample['bits2']))
        params_list.append(sample['params'])
    
    # Stack 所有 tensor
    return {
        'mixsignal_ri': torch.stack(mixsignal_list, dim=0),  # (B, 2, max_len)
        'rfsignal1_ri': torch.stack(rfsignal1_list, dim=0),  # (B, 2, max_len)
        'rfsignal2_ri': torch.stack(rfsignal2_list, dim=0),  # (B, 2, max_len)
        'bits1': bits1_list,  # List of tensors (不同长度)
        'bits2': bits2_list,  # List of tensors (不同长度)
        'params': params_list,  # List of strings
        'original_lengths': torch.tensor(original_lengths, dtype=torch.long),  # (B,)
    }

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser("DDP Inference for SignalSeparator (multi-mod, CSV only)")
    # parser.add_argument('--ckpt_path', type=str, required=True)
    # parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results/infer')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='use torch.cuda.amp for inference')
    args = parser.parse_args()
    config = load_config(os.sep.join(['configs', args.config]))

    setup_ddp(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank, world = get_rank(), get_world_size()

    # 读取数据 & DDP 切分
    # loaded_data = torch.load(args.test_data_path)  # list of dict
    # _, dataset = get_decode_train_test_100k(m=1, base=('/nas/datasets/zjw/PCMA_8psk/diffusion_prediction_ddnm/'))
    print(config.data.decoding_files)
    dataset = PCMAModDataset_Decoding_Generate(config.data.decoding_files, signal_len=config.data.signal_len, modulation=config.data.modulation)
    # dataset = GenericTestDataset(loaded_data)
    # sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                #  shuffle=False, drop_last=False) if dist_is_initialized() else None
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        )

    TMP_DIR = os.path.join(args.out_dir, "_tmp_csv")
    if rank == 0:
        os.makedirs(TMP_DIR, exist_ok=True)

    results = []
    autocast_enabled = (device.type == 'cuda') and args.amp
    cnt_1 = 0
    cnt_correct_1 = 0
    cnt_2 = 0
    cnt_correct_2 = 0
    snr1_all = []
    snr2_all = []
    all_num = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            print(bi)
            all_num += len(batch)
            # if bi * 64 > 10000 or bi * 64 <= 7500:
            #     continue
            # x = batch['data'][''].to(device)  # (B,2,T)

            # if autocast_enabled:
            #     with torch.cuda.amp.autocast():
            #         y = model(x)
            # else:
            #     y = model(x)

            # if isinstance(y, (tuple, list)):
            #     # 模型返回 list[4], 每个 [B,1,T] -> [B,4,T]
            #     y = torch.cat(y, dim=1)

            # y_np = y.detach().cpu().numpy()
            p1 = batch['data']['rf_signal1_predict'].cpu().numpy()
            p2 = batch['data']['rf_signal2_predict'].cpu().numpy()
            mix = batch['mix'].cpu().numpy()
            # p2 = mix - p1
            # print(p1.shape)
            # p1 = torch.stack([p1.real, p1.imag], dim=1).to(torch.float32)
            # p2 = torch.stack([p2.real, p2.imag], dim=1).to(torch.float32)

            # p1 = (p1 - p1.mean())/p1.std()
            # p2 = (p2 - p2.mean())/p2.std()

            g1 = batch['data']['rfsignal1'].cpu().numpy()
            g2 = batch['data']['rfsignal2'].cpu().numpy()
            snr1_all.append(batch_snr_db(p1, g1))
            snr2_all.append(batch_snr_db(p2, g2))
            # p1 = g1
            # p2 = g2
            # print(snr1_all)
            # p1 = g1 / g1.std()
            # p2 = g2 / g2.std()
            # print(g1.shape)
            # std = 0.0
            # p1 = g1 + std * np.linalg.norm(g1, axis=(1,2), keepdims=True) / g1.shape[2]**0.5 * np.random.randn(*g1.shape)
            # p2 = g2 + std * np.linalg.norm(g2, axis=(1,2), keepdims=True) / g2.shape[2]**0.5 * np.random.randn(*g2.shape)
            # p1 = g2
            # p2 = g1
            # original_lengths = batch['original_lengths'].cpu().numpy()  # (B,)
            

            # 归一化 MSE（只计算原始长度部分）
            B = p1.shape[0]
            loss1_list = []
            loss2_list = []
            original_lengths = [3072 for _ in range(B)]
            for i in range(B):
                orig_len = int(original_lengths[i])
                p1_i = p1[i, :, :orig_len]
                p2_i = p2[i, :, :orig_len]
                g1_i = g1[i, :, :orig_len]
                g2_i = g2[i, :, :orig_len]
                loss1_i = ((p1_i - g1_i) ** 2).mean() / (np.linalg.norm(g1_i) + 1e-12)
                loss2_i = ((p2_i - g2_i) ** 2).mean() / (np.linalg.norm(g2_i) + 1e-12)
                loss1_list.append(loss1_i)
                loss2_list.append(loss2_i)
            loss1 = np.array(loss1_list)
            loss2 = np.array(loss2_list)

            for i in range(B):
                orig_len = int(original_lengths[i])
                # 截断到原始长度
                pr1 = (p1[i,0,:orig_len] + 1j*p1[i,1,:orig_len]).astype(np.complex64)
                pr2 = (p2[i,0,:orig_len] + 1j*p2[i,1,:orig_len]).astype(np.complex64)
                gt1 = (g1[i,0,:orig_len] + 1j*g1[i,1,:orig_len]).astype(np.complex64)
                gt2 = (g2[i,0,:orig_len] + 1j*g2[i,1,:orig_len]).astype(np.complex64)

                params_batch = batch['data']['params']
                # print(len(params_batch[0]))
                # if isinstance(params_batch, (list, tuple)):
                #     params_i = params_batch[i]
                # else:
                #     params_i = str(params_batch)
                params_i = [params_batch[n][i] for n in range(len(params_batch))]
                meta = parse_params_all(str(params_i))
                # print(meta)
                delay1 = meta['delay1']
                delay2 = meta['delay2']
                # if abs(delay1 - delay2) < 2:
                #     continue

                f1 = meta.get('f_off1', 0.0) or 0.0
                f2 = meta.get('f_off2', 0.0) or 0.0
                phi1 = meta.get('phi1', 0.0) or 0.0
                phi2 = meta.get('phi2', 0.0) or 0.0
                mod1 = meta.get('mod1') or "QPSK"
                mod2 = meta.get('mod2') or "QPSK"
                # print(mod1)
                # print(mod2)

                n = np.arange(len(pr1))
                t = n / fs

                # # 理想补偿 CFO + 相位
                # pr1_c = pr1 * np.exp(-1j * (2 * np.pi * float(f1) * t + float(phi1)))
                # pr2_c = pr2 * np.exp(-1j * (2 * np.pi * float(f2) * t + float(phi2)))
                # gt1_c = gt1 * np.exp(-1j * (2 * np.pi * float(f1) * t + float(phi1)))
                # gt2_c = gt2 * np.exp(-1j * (2 * np.pi * float(f2) * t + float(phi2)))

                # # MF + 抽样 + guard + 幅度归一化
                # ps1 = mf_and_sample(pr1_c, sps, rc, num_taps)
                # ps2 = mf_and_sample(pr2_c, sps, rc, num_taps)
                # gs1 = mf_and_sample(gt1_c, sps, rc, num_taps)
                # gs2 = mf_and_sample(gt2_c, sps, rc, num_taps)

                ps1 = mf_and_sample(pr1, sps, rc, num_taps)
                n_sym = np.arange(len(ps1))
                t_sym = n_sym * sps / fs
                ps1 = ps1 * np.exp(-1j*(2*np.pi*f1*t_sym + phi1))

                ps2 = mf_and_sample(pr2, sps, rc, num_taps)
                n_sym = np.arange(len(ps2))
                t_sym = n_sym * sps / fs
                ps2 = ps2 * np.exp(-1j*(2*np.pi*f2*t_sym + phi2))

                gs1 = mf_and_sample(gt1, sps, rc, num_taps)
                n_sym = np.arange(len(gs1))
                t_sym = n_sym * sps / fs
                gs1 = gs1 * np.exp(-1j*(2*np.pi*f1*t_sym + phi1))

                gs2 = mf_and_sample(gt2, sps, rc, num_taps)
                n_sym = np.arange(len(gs2))
                t_sym = n_sym * sps / fs
                gs2 = gs2 * np.exp(-1j*(2*np.pi*f2*t_sym + phi2))

                L1 = min(len(ps1), len(gs1))
                L2 = min(len(ps2), len(gs2))
                
                if L1 <= 0 or L2 <= 0:
                    print(f"L1: {L1}, L2: {L2}")
                    print(f"ps1: {ps1.shape}, gs1: {gs1.shape}")
                    print(f"ps2: {ps2.shape}, gs2: {gs2.shape}")
                    print(f"pr1: {pr1.shape}, gt1: {gt1.shape}")
                    print(f"pr2: {pr2.shape}, gt2: {gt2.shape}")
                    print(f"f1: {f1}, f2: {f2}")
                    print(f"phi1: {phi1}, phi2: {phi2}")
                    print(f"mod1: {mod1}, mod2: {mod2}")
                    continue
                ps1 = ps1[:L1]; gs1 = gs1[:L1]
                ps2 = ps2[:L2]; gs2 = gs2[:L2]

                # 相位对齐
                # ps1_aligned = align_phase(gs1, ps1)
                # ps2_aligned = align_phase(gs2, ps2)
                step = 2*np.pi/8
                ps1_aligned = ps1 * np.exp(-1j * 0 * step)
                ps2_aligned = ps2 * np.exp(-1j * 0 * step)
                


                evm1 = evm_rms(gs1, ps1_aligned)
                evm2 = evm_rms(gs2, ps2_aligned)

                # bits 参考（bits1/bits2 是 list of tensors）
                b1_full = batch['bits1'][i].cpu().numpy().astype(np.int8)
                b2_full = batch['bits2'][i].cpu().numpy().astype(np.int8)

                bps1 = BITS_PER_SYMBOL.get(mod1.upper(), 2)
                bps2 = BITS_PER_SYMBOL.get(mod2.upper(), 2)

                b1_ref = slice_bits_to_match_syms(b1_full, len(gs1), bps1)
                b2_ref = slice_bits_to_match_syms(b2_full, len(gs2), bps2)

                # 解调
                # print(mod1)
                # print(mod2)
                b1_hat = demod_by_mod(ps1_aligned, mod1)
                b2_hat = demod_by_mod(ps2_aligned, mod2)
                # def best_bit_shift_ber(b_hat, b_ref, max_sym_shift=2, bps=3):
                #     best = 1.0
                #     for s in range(-max_sym_shift, max_sym_shift+1):
                #         shift = s * bps
                #         if shift >= 0:
                #             err = np.mean(b_hat[shift:] != b_ref[:len(b_hat)-shift])
                #         else:
                #             err = np.mean(b_hat[:shift] != b_ref[-shift:])
                #         best = min(best, err)
                #     return best
                # ber1 = best_bit_shift_ber(b1_hat, b1_ref, max_sym_shift=2, bps=bps1)
                # ber2 = best_bit_shift_ber(b1_hat, b1_ref, max_sym_shift=2, bps=bps1)

                # print(b1_hat[0:100])
                # print(b1_ref[0:100])
                # print('===================')
                ber1, b1_hat_aligned, b1_ref_aligned, shift1 = \
                    align_bits_and_compute_ber(
                        b1_hat,
                        b1_ref,
                        bits_per_sym=bps1,
                        max_sym_shift=8,
                    )

                ber2, b2_hat_aligned, b2_ref_aligned, shift2 = \
                    align_bits_and_compute_ber(
                        b2_hat,
                        b2_ref,
                        bits_per_sym=bps2,
                        max_sym_shift=8,
                    )
                b1_hat = b1_hat_aligned
                b2_hat = b2_hat_aligned
                b1_ref = b1_ref_aligned
                b2_ref = b2_ref_aligned
                Lb1 = min(len(b1_hat), len(b1_ref))
                Lb2 = min(len(b2_hat), len(b2_ref))
                for k in range(len(b1_hat)//3):
                    cnt_1 += 1
                    if b1_hat[3*k] == b1_ref[3*k] and b1_hat[3*k + 1] == b1_ref[3*k + 1] and b1_hat[3*k + 2] == b1_ref[3*k + 2]:
                        cnt_correct_1 += 1
                for k in range(len(b2_hat)//3):
                    cnt_2 += 1
                    if b2_hat[3*k] == b2_ref[3*k] and b2_hat[3*k + 1] == b2_ref[3*k + 1] and b2_hat[3*k + 2] == b2_ref[3*k + 2]:
                        cnt_correct_2 += 1
                
                ber1 = float(np.mean(b1_hat[:Lb1] != b1_ref[:Lb1])) if Lb1 > 0 else 1.0
                ber2 = float(np.mean(b2_hat[:Lb2] != b2_ref[:Lb2])) if Lb2 > 0 else 1.0
                ber  = 0.5 * (ber1 + ber2)
                ser2 = 1-(cnt_correct_2) / float(cnt_2)
                ser1 = 1-(cnt_correct_1) / float(cnt_1)
                # if ber2 > 0:
                #     print(b2_hat[0:40])
                #     print(b2_ref[0:40])
                #     print('===============================================')
                # print('')
                phi_diff = float(wrap_2pi(phi2 - phi1))

                results.append({
                    'loss1': float(loss1[i]),
                    'loss2': float(loss2[i]),
                    'BER1': ber1, 'BER2': ber2, 'BER': ber,
                    'snr': meta.get('snr'),
                    'amp': meta.get('amp'),
                    'f1': meta.get('f_off1'),
                    'f2': meta.get('f_off2'),
                    'phi1': phi1,
                    'phi2': phi2,
                    'phi_diff': phi_diff,
                    'rep': meta.get('rep'),
                    'delay1': meta.get('delay1'),
                    'delay2': meta.get('delay2'),
                    'delay_diff': meta.get('delay_diff'),
                    'mod1': mod1,
                    'mod2': mod2,
                    'evm1': evm1,
                    'evm2': evm2,
                    'ser2': ser2,
                })

            if rank == 0:
                print(f"[Rank0][Batch {bi}] done. BER1: {ber1}, BER2: {ber2}, BER: {ber}, SER1: {ser1}, SER2: {ser2}")
            # break
        snr1_all = np.concatenate(snr1_all, axis=0)
        snr2_all = np.concatenate(snr2_all, axis=0)

        snr1_mean = snr1_all.mean()
        snr2_mean = snr2_all.mean()
        print(f"SER1: {ser1}, SER2: {ser2}")

        print(f"SNR1 mean: {snr1_mean:.2f} dB")
        print(f"SNR2 mean: {snr2_mean:.2f} dB")
    # 每个 rank 写各自的 csv
    tmp_csv = os.path.join(TMP_DIR, f"metrics_rank{rank}.csv")
    os.makedirs(TMP_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(tmp_csv, index=False)

    # 同步并合并
    if dist_is_initialized():
        dist.barrier()

    if rank == 0:
        dfs = []
        for r in range(world):
            path = os.path.join(TMP_DIR, f"metrics_rank{r}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True) if len(dfs) else pd.DataFrame()
        final_csv = os.path.join(args.out_dir, "metrics_all.csv")
        df.to_csv(final_csv, index=False)
        print(f"[Rank0] merged metrics saved to: {final_csv}")
        if "BER" in df.columns and len(df):
            print("Overall mean BER:", df["BER"].mean())
        else:
            print("Overall mean BER: N/A")

    cleanup_ddp()

if __name__ == '__main__':
    main()
