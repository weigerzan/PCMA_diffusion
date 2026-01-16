"""
解调测试脚本：对模型输出和ground truth进行解调并计算误符号率（SER）

功能：
1. 每次只处理一个切片（pth文件中的一个条目）
2. 支持从单个.npy文件加载模型输出和ground truth两个信号
3. 也支持分别从.npy文件和pth文件加载pred和gt
4. 对两个结果都进行解调
5. 计算误符号率（SER）
6. 支持只输入ground truth进行测试（从test的pth文件中读取单个切片）

使用示例：
  # 方式1：从单个npy文件加载pred和gt（推荐）
  python final_test/test_demodulation_ser.py \
    --pred_file /path/to/pred_and_gt.npy \
    --modulation 8PSK \
    --output_dir ./results
  # npy文件格式支持：
  #   - 形状为 (2, N) 的数组：第一行是pred，第二行是gt
  #   - 形状为 (N, 2) 的数组：第一列是pred，第二列是gt
  #   - 字典：包含 'pred'/'gt' 或 'pred_signal'/'gt_signal' 键
  #   - 元组/列表：长度为2，第一个是pred，第二个是gt

  # 方式2：分别从npy和pth文件加载
  python final_test/test_demodulation_ser.py \
    --pred_file /path/to/pred_signal1.npy \
    --gt_file /path/to/test_shard.pth \
    --modulation 8PSK \
    --signal_idx 0 \
    --entry_idx 0 \
    --output_dir ./results

  # 方式3：只测试ground truth（从pth文件读取单个切片）
  python final_test/test_demodulation_ser.py \
    --gt_file /nas/datasets/yixin/PCMA/real_data/8psk/test/real_8psk_mixed_amp0.2to1.0_shard00_of03_c128.pth \
    --modulation 8PSK \
    --signal_idx 0 \
    --entry_idx 0 \
    --output_dir ./results \
    --gt_only
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import numpy as np
import torch
import tqdm

import multiprocessing as mp
from functools import partial
import tqdm
import os


from decoder_real.test_utils import (
    FS, SPS, BETA, NUM_TAPS,
    demodulate_real_signal,
    evaluate_clustering_quality,
    calculate_ser_from_symbols
)
from datasets.decoding_real_dataset import PCMAModDataset_Decoding_Generate_Real
from utils.config import load_config


USE_CONSISTENT_DEMOD = False

FS = 12e6
SPS = 8
BETA = 0.33
NUM_TAPS = 64
# NUM_process = 64

def process_one_item(params_list):
    n, data, args, output_dir = params_list
    pred_signal, gt_signal = load_two_signals_from_npy(data)

    # ------ ground truth 加载逻辑（原样复制） ------
    if gt_signal is None:
        if args.gt_file is None:
            if args.gt_only:
                raise ValueError("gt_only模式需要提供--gt_file参数")
            elif pred_signal is None:
                raise ValueError("需要提供--pred_file或--gt_file参数")
            else:
                raise ValueError("pred_file中未找到gt信号，需要提供--gt_file参数")

        if args.gt_only and pred_signal is not None:
            print("警告: gt_only模式下忽略pred_signal")
            pred_signal = None

        try:
            gt_signal, mod_from_file = load_slice_from_pth(
                args.gt_file, signal_idx=args.signal_idx, entry_idx=args.entry_idx
            )
        except Exception as e:
            print(f"加载GT失败: {e}")
            return None

    if args.gt_only and pred_signal is not None:
        pred_signal = None

    # ------ SNR ------
    snr1 = batch_snr_db(pred_signal[0:1], gt_signal[0:1])
    snr2 = batch_snr_db(pred_signal[1:2], gt_signal[1:2])

    auto_search = not args.no_auto_search

    # ------ 调用 test_single_pair ------
    result1 = test_single_pair(
        args,
        pred_signal[0], gt_signal[0], args.modulation,
        loop_bandwidth=args.loop_bandwidth,
        auto_search=auto_search
    )
    result2 = test_single_pair(
        args,
        pred_signal[1], gt_signal[1], args.modulation,
        loop_bandwidth=args.loop_bandwidth,
        auto_search=auto_search
    )

    # ------ 保存结果文件（每个进程写自己文件） ------
    result_file = output_dir / f"demod_result_{n}.json"
    with open(result_file, 'w') as f:
        json.dump({
            "result1": result1,
            "result2": result2,
        }, f, indent=2)

    # ------ 返回子结果供主进程收集 ------
    return {
        "snr1": snr1,
        "snr2": snr2,
        "ber1": result1["ser"],
        "ber2": result2["ser"],
        "n": n
    }


# ------------------------------
# 主程序：调用多进程并行
# ------------------------------
def run_all(data, args, output_dir, num_workers=8):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_list = list(range(0, len(data), 10))

    # 组装 worker 部分参数
    worker = partial(process_one_item, data=data, args=args, output_dir=output_dir)

    snr_signal_1 = []
    snr_signal_2 = []
    ber_signal_1 = []
    ber_signal_2 = []
    params_list = [(n, data[n], args, output_dir) for n in index_list]

    with mp.Pool(num_workers) as pool:
        # for res in tqdm.tqdm(pool.imap(worker, index_list), total=len(index_list)):
        for res in pool.imap(process_one_item, params_list):

            if res is None:   # 某任务失败
                continue

            snr_signal_1.append(res["snr1"])
            snr_signal_2.append(res["snr2"])
            ber_signal_1.append(res["ber1"])
            ber_signal_2.append(res["ber2"])

    # 最终返回或保存（你自行决定）
    return snr_signal_1, snr_signal_2, ber_signal_1, ber_signal_2

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
    # assert p.shape == g.shape
    # assert p.ndim == 3 and p.shape[1] == 2

    # IQ -> complex: (B, T)
    # p_c = p[:, 0, :] + 1j * p[:, 1, :]
    # g_c = g[:, 0, :] + 1j * g[:, 1, :]
    p_c = p
    g_c = g

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

def load_signal_from_npy(file_path: str) -> np.ndarray:
    """
    从.npy文件加载信号
    
    Args:
        file_path: .npy文件路径
        
    Returns:
        复数信号数组
    """
    signal = np.load(file_path)
    signal = np.asarray(signal, dtype=np.complex128)
    return signal


def load_two_signals_from_npy(data) -> Tuple[np.ndarray, np.ndarray]:
    """
    从.npy文件加载两个信号（模型输出和ground truth）
    
    支持多种格式：
    1. 形状为 (2, N) 的数组：第一行是pred，第二行是gt
    2. 形状为 (N, 2) 的数组：第一列是pred，第二列是gt
    3. 字典：包含 'pred' 和 'gt' 键
    4. 元组/列表：长度为2，第一个是pred，第二个是gt
    
    Args:
        file_path: .npy文件路径
        
    Returns:
        (pred_signal, gt_signal)
    """
    # data = np.load(file_path, allow_pickle=True)
    
    # 如果是字典
    # if isinstance(data, dict):
    #     if 'pred' in data and 'gt' in data:
    #         pred = np.asarray(data['data']['rf_signal1_predict'], dtype=np.complex128)
    #         gt = np.asarray(data['gt'], dtype=np.complex128)
    #         return pred, gt
    #     elif 'pred_signal' in data and 'gt_signal' in data:
    #         pred = np.asarray(data['pred_signal'], dtype=np.complex128)
    #         gt = np.asarray(data['gt_signal'], dtype=np.complex128)
    #         return pred, gt
    #     else:
    #         raise ValueError(f"字典中找不到 'pred'/'gt' 或 'pred_signal'/'gt_signal' 键，可用键: {list(data.keys())}")
    p1 = data['rf_signal1_predict']
    p2 = data['rf_signal2_predict']
    # print(p1)
    # print(p2)
    # pred1 = p1[0:1] + 1j * p1[1:2]
    # pred2 = p2[0:1] + 1j * p2[1:2]
    # 给这两个numpy数组都加一个维度，然后拼接
    p1 = np.expand_dims(p1, axis=0)
    p2 = np.expand_dims(p2, axis=0)
    pred = np.concatenate([p1, p2], axis=0)
    # mix = data['mix'].cpu().numpy()
    # p2 = mix - p1
    # print(p1.shape)
    # p1 = torch.stack([p1.real, p1.imag], dim=1).to(torch.float32)
    # p2 = torch.stack([p2.real, p2.imag], dim=1).to(torch.float32)

    # p1 = (p1 - p1.mean())/p1.std()
    # p2 = (p2 - p2.mean())/p2.std()

    g1 = data['rfsignal1']
    g2 = data['rfsignal2']
    # 给这两个numpy数组都加一个维度，然后拼接
    g1 = np.expand_dims(g1, axis=0)
    g2 = np.expand_dims(g2, axis=0)
    # gt1 = g1[0:1] + 1j * g1[1:2]
    # gt2 = g2[0:1] + 1j * g2[1:2]
    gt = np.concatenate([g1, g2], axis=0)
    return pred, gt
    # # 如果是元组或列表
    # if isinstance(data, (tuple, list)) and len(data) == 2:
    #     pred = np.asarray(data[0], dtype=np.complex128)
    #     gt = np.asarray(data[1], dtype=np.complex128)
    #     return pred, gt
    
    # # 如果是numpy数组
    # data = np.asarray(data)
    
    # # 形状为 (2, N) 或 (2, N, ...)
    # if data.shape[0] == 2:
    #     pred = np.asarray(data[0], dtype=np.complex128)
    #     gt = np.asarray(data[1], dtype=np.complex128)
    #     return pred, gt
    
    # # 形状为 (N, 2) 或 (N, 2, ...)
    # if len(data.shape) >= 2 and data.shape[1] == 2:
    #     pred = np.asarray(data[:, 0], dtype=np.complex128)
    #     gt = np.asarray(data[:, 1], dtype=np.complex128)
    #     return pred, gt
    
    # # 如果无法识别格式，抛出错误
    # raise ValueError(f"无法识别数据格式。形状: {data.shape}, 类型: {type(data)}。"
    #                  f"请确保数据格式为 (2, N), (N, 2), 字典或长度为2的元组/列表")


def load_slice_from_pth(pth_file: str, signal_idx: int = 0, entry_idx: int = 0) -> Tuple[np.ndarray, Optional[str]]:
    """
    从pth文件加载单个切片的信号（ground truth）
    
    Args:
        pth_file: pth文件路径
        signal_idx: 信号索引（0=rfsignal1, 1=rfsignal2）
        entry_idx: 条目索引（pth文件中的第几个切片，默认0）
        
    Returns:
        (信号数组, 调制方式)
    """
    data = torch.load(pth_file, map_location='cpu')
    
    # 获取指定索引的条目
    if isinstance(data, list):
        if entry_idx >= len(data):
            raise ValueError(f"条目索引 {entry_idx} 超出范围（共 {len(data)} 个条目）")
        entry = data[entry_idx]
    elif isinstance(data, dict):
        # 如果是单个字典，直接使用
        entry = data
    else:
        raise ValueError(f"不支持的pth文件格式: {type(data)}")
    
    # 提取信号
    if signal_idx == 0:
        signal_key = 'rfsignal1'
    elif signal_idx == 1:
        signal_key = 'rfsignal2'
    else:
        raise ValueError(f"signal_idx 必须是 0 或 1，当前为 {signal_idx}")
    
    if signal_key not in entry:
        raise ValueError(f"pth文件中找不到 {signal_key}，可用字段: {list(entry.keys())}")
    
    signal = np.asarray(entry[signal_key], dtype=np.complex128)
    
    # 尝试从params中提取调制方式
    modulation = None
    if 'params' in entry:
        params = entry['params']
        if isinstance(params, (list, tuple)) and len(params) > 9:
            # params格式: (snr_db, amplitude_ratio, sps, f_off1_str, f_off2_str, phi1_str, phi2_str, delay1_str, delay2_str, mod1_str, mod2_str, ...)
            mod_str = params[9] if signal_idx == 0 else params[10]
            if isinstance(mod_str, str) and mod_str.startswith('mod'):
                modulation = mod_str.split('=')[1] if '=' in mod_str else None
    
    return signal, modulation



def demodulate_signal(signal: np.ndarray, modulation: str, 
                      loop_bandwidth: Optional[float] = None,
                      auto_search: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """
    对信号进行解调：Costas环补偿 + 匹配滤波 + 符号抽取
    
    使用test_utils.py中的demodulate_real_signal函数，确保与生成数据版本一致
    
    Args:
        signal: 输入复数信号
        modulation: 调制方式
        loop_bandwidth: Costas环带宽（如果为None且auto_search=True，则自动搜索）
        auto_search: 是否自动搜索最优loop_bandwidth
        
    Returns:
        (symbols, score, info_dict)
        - symbols: 解调得到的符号
        - score: 聚类质量得分
        - info_dict: 包含解调信息的字典（loop_bandwidth, offset等）
    """
    # 直接使用test_utils.py中的函数，确保与生成数据版本一致
    symbols, score, info_dict = demodulate_real_signal(
        signal, modulation,
        loop_bandwidth=loop_bandwidth,
        auto_search=auto_search,
        threshold=6.0
    )
    
    return symbols, score, info_dict


def hard_decide_symbols(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    硬判决：映射到最近的理想星座点索引
    
    Args:
        symbols: 复数符号数组
        modulation: 调制方式
        
    Returns:
        符号索引数组
    """
    modulation = modulation.upper()
    
    if modulation == 'QPSK':
        M = 4
        phases = 2 * np.pi * np.arange(M) / M
        ideal_constellation = np.exp(1j * phases)
    elif modulation == '8PSK':
        M = 8
        phases = 2 * np.pi * np.arange(M) / M
        ideal_constellation = np.exp(1j * phases)
    elif modulation == '16QAM':
        levels = np.array([-3, -1, 1, 3], dtype=float)
        xv, yv = np.meshgrid(levels, levels)
        ideal_constellation = xv.flatten() + 1j * yv.flatten()
        ideal_constellation = ideal_constellation / np.sqrt(10.0)
    else:
        raise ValueError(f"不支持的调制方式: {modulation}")
    
    symbols = np.asarray(symbols, dtype=np.complex128)
    # 归一化
    power = np.mean(np.abs(symbols)**2)
    if power == 0:
        return np.zeros(len(symbols), dtype=np.int32)
    symbols_norm = symbols / np.sqrt(power)
    
    # 归一化理想点
    ideal_power = np.mean(np.abs(ideal_constellation)**2)
    if ideal_power > 0:
        ideal_constellation = ideal_constellation / np.sqrt(ideal_power)
    
    # 计算距离并找到最近的星座点
    diff = symbols_norm[:, np.newaxis] - ideal_constellation[None, :]
    dist2 = np.abs(diff) ** 2
    decision_indices = np.argmin(dist2, axis=1)
    
    return decision_indices


def calculate_ser(symbols_ref: np.ndarray, symbols_pred: np.ndarray, modulation: str) -> float:
    """
    计算误符号率（Symbol Error Rate）
    
    Args:
        symbols_ref: 参考符号（ground truth解调）
        symbols_pred: 预测符号（模型输出解调）
        modulation: 调制方式
        
    Returns:
        SER: 误符号率（0~1）
    """
    # 硬判决
    decided_ref = hard_decide_symbols(symbols_ref, modulation)
    decided_pred = hard_decide_symbols(symbols_pred, modulation)
    
    # 对齐长度
    min_len = min(len(decided_ref), len(decided_pred))
    if min_len == 0:
        return float('nan')
    
    decided_ref = decided_ref[:min_len]
    decided_pred = decided_pred[:min_len]
    
    # 计算错误数
    errors = np.sum(decided_ref != decided_pred)
    ser = errors / min_len
    
    return ser


def test_single_pair(args, pred_signal: Optional[np.ndarray], 
                     gt_signal: np.ndarray,
                     modulation: str,
                     loop_bandwidth: Optional[float] = None,
                     auto_search: bool = True) -> Dict:
    """
    测试单个信号对的解调和SER
    
    Args:
        pred_signal: 模型输出信号（如果为None，则只测试ground truth）
        gt_signal: ground truth信号
        modulation: 调制方式
        loop_bandwidth: Costas环带宽（可选）
        auto_search: 是否自动搜索最优loop_bandwidth
        
    Returns:
        结果字典
    """
    # 解调ground truth
    if not args.disable_middle_output:
        print("  解调ground truth信号...")
    gt_symbols, gt_score, gt_info = demodulate_signal(
        gt_signal, modulation, loop_bandwidth=loop_bandwidth, auto_search=auto_search
    )
    if not args.disable_middle_output:
        print(f"    GT符号数: {len(gt_symbols)}, 聚类得分: {gt_score:.4f}")
    
    result = {
        'gt_num_symbols': len(gt_symbols),
        'gt_clustering_score': float(gt_score),
        'gt_info': gt_info
    }
    
    # 如果提供了预测信号，进行对比
    if pred_signal is not None:
        if not args.disable_middle_output:
            print("  解调模型输出信号...")
        pred_symbols, pred_score, pred_info = demodulate_signal(
            pred_signal, modulation, loop_bandwidth=loop_bandwidth, auto_search=auto_search
        )
        if not args.disable_middle_output:
            print(f"    Pred符号数: {len(pred_symbols)}, 聚类得分: {pred_score:.4f}")
        
        # 计算SER
        # ser = calculate_ser(gt_symbols, pred_symbols, modulation)
        ser = calculate_ser_from_symbols(gt_symbols, pred_symbols, modulation)
        if not args.disable_middle_output:
            print(f"    SER: {ser:.6f}")
        
        result.update({
            'pred_num_symbols': len(pred_symbols),
            'pred_clustering_score': float(pred_score),
            'pred_info': pred_info,
            'ser': float(ser)
        })
    else:
        result['ser'] = None
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="解调测试脚本：对模型输出和ground truth进行解调并计算SER"
    )
    parser.add_argument("--disable_middle_output", action="store_true",
                       help="模型输出信号文件")
    parser.add_argument("--loop_bandwidth", type=float, default=None,
                       help="Costas环带宽（如果为None，则自动搜索）")
    parser.add_argument("--no_auto_search", action="store_true",
                       help="禁用自动搜索loop_bandwidth（使用默认值或指定值）")
    parser.add_argument("--output_dir", type=str, default="./demod_test_results",
                       help="输出目录")
    parser.add_argument("--gt_only", action="store_true",
                       help="只测试ground truth（不加载pred_file）")
    parser.add_argument('--config', type=str, required=True)
    
    
    args = parser.parse_args()

    

    config = load_config(os.sep.join(['configs', args.config]))
    args.modulation = config.data.modulation 
    
    # 确定输出目录：如果使用默认值，则使用模型目录下的demod_results子目录
    if args.output_dir == "./demod_test_results":
        # 使用模型输出目录下的demod_results子目录
        model_output_dir = Path(config.training.output_dir)
        output_dir = model_output_dir / "demod_results"
    else:
        output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"解调结果将保存到: {output_dir}")
    
    print("=" * 70)
    print("解调测试脚本")
    print("=" * 70)
    print(f"调制方式: {args.modulation}")
    # print(f"信号索引: {args.signal_idx} ({'rfsignal1' if args.signal_idx == 0 else 'rfsignal2'})")
    
    # 加载信号
    pred_signal = None
    gt_signal = None
    
    # print('loading data from {}'.format(args.pred_file))
    # data = torch.load(args.pred_file, weights_only=False)
    dataset = PCMAModDataset_Decoding_Generate_Real(file_list=config.data.decoding_files)
    # print('data loaded.')
    ber_signal_1 = []
    ber_signal_2 = []
    snr_signal_1 = []
    snr_signal_2 = []
    snr_signal_1, snr_signal_2, ber_signal_1, ber_signal_2 = run_all(
        dataset,
        args,
        output_dir=str(output_dir),  # 使用确定的输出目录
        num_workers=128
    )
    avg_ber_1 = np.nanmean(ber_signal_1)
    avg_ber_2 = np.nanmean(ber_signal_2)
    avg_snr_1 = np.nanmean(snr_signal_1)
    avg_snr_2 = np.nanmean(snr_signal_2)
    print(f'\n平均SER signal 1: {avg_ber_1:.6f}')
    print(f'平均SER signal 2: {avg_ber_2:.6f}')
    print(f'平均SER: {(avg_ber_1 + avg_ber_2)/2:.6f}')
    print(f'\n平均SNR signal 1: {avg_snr_1:.6f} dB')
    print(f'平均SNR signal 2: {avg_snr_2:.6f} dB')
    print(f'平均SNR: {(avg_snr_1 + avg_snr_2)/2:.6f} dB')


if __name__ == "__main__":
    main()

