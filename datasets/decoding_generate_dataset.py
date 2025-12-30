import torch
from torch.utils.data import Dataset, Subset
import glob
import os
import tqdm
# from torch.utils.data import 
from .base_dataset import PCMAModDataset
import re
import numpy as np

fs = 12e6

class PCMAModDataset_Decoding_Generate(PCMAModDataset):
    def __init__(self, file_list, transform=None):
        super().__init__(file_list=file_list, transform=transform)
        self.m = 1
        self.bits_per_block = int(384*3 / self.m)
        self.samples_per_block = int(3072 / self.m)
        self.symbols_per_block = int(384 / self.m)

    def __len__(self):
        return super().__len__() * self.m

    def __getitem__(self, idx):
        shard_idx = idx // self.m
        shard_id = int(torch.searchsorted(self.cum_sizes, shard_idx, right=True))
        shard_start = 0 if shard_id == 0 else int(self.cum_sizes[shard_id - 1])
        local_idx = shard_idx - shard_start

        self._load_shard(shard_id)
        data = self.current_data[local_idx]
        # 'mixsignal'为混合数据，'rfsignal1'和'rfsignal2'为对应干净数据，都是torch.complex64
        mix = torch.from_numpy(data['mixsignal'])
        # print(data['params'])
        # values = [float(re.search(r'[-+]?\d*\.?\d+', x).group()) for x in data['params']]
        clean1 = data['rf_signal1_predict']
        clean2 = data['rf_signal2_predict']
        phi1 = float(re.search(r'(\-?\d+\.?\d*)', data['params'][5].split('=')[-1]).group())
        phi2 = float(re.search(r'(\-?\d+\.?\d*)', data['params'][6].split('=')[-1]).group())
        f_off1 = float(re.search(r'(\-?\d+\.?\d*)', data['params'][3].split('=')[-1]).group())
        f_off2 = float(re.search(r'(\-?\d+\.?\d*)', data['params'][4].split('=')[-1]).group())

        clean1, clean2 = align_phase(f_off1, f_off2, phi1, phi2, clean1, clean2)
        clean1 = torch.from_numpy(clean1)
        clean2 = torch.from_numpy(clean2)
        bits1 = torch.from_numpy(data['bits1'])
        bits2 = torch.from_numpy(data['bits2'])
        # 转换为3072 * 2
        mix = torch.stack([mix.real, mix.imag], dim=0)  # [2, 3072]
        clean1 = torch.stack([clean1.real, clean1.imag], dim=0).to(torch.float32)
        clean2 = torch.stack([clean2.real, clean2.imag], dim=0).to(torch.float32)
        # TODO 归一化怎么做？暂定对mix做mean-var归一化
        idx_in_block = idx % self.m
        clean1 = clean1[:, idx_in_block * self.samples_per_block : (idx_in_block + 1) * self.samples_per_block]
        clean2 = clean2[:, idx_in_block * self.samples_per_block : (idx_in_block + 1) * self.samples_per_block]
        bits1 = bits1[idx_in_block * self.bits_per_block : (idx_in_block + 1) * self.bits_per_block]
        bits2 = bits2[idx_in_block * self.bits_per_block : (idx_in_block + 1) * self.bits_per_block]
        labels1 = torch.zeros(self.symbols_per_block, dtype=torch.long)
        labels2 = torch.zeros(self.symbols_per_block, dtype=torch.long)
        for i in range(self.symbols_per_block):
            bits_segment1 = bits1[i*3:(i+1)*3]
            bits_segment2 = bits2[i*3:(i+1)*3]
            label1 = bits_segment1[0] * 4 + bits_segment1[1] * 2 + bits_segment1[2] * 1
            label2 = bits_segment2[0] * 4 + bits_segment2[1] * 2 + bits_segment2[2] * 1
            labels1[i] = label1
            labels2[i] = label2
        # clean1 = (clean1 - clean1.mean())/clean1.std()
        # clean2 = (clean2 - clean2.mean())/clean2.std()
        data['rf_signal1_predict'] = torch.from_numpy(data['rf_signal1_predict'])
        data['rf_signal2_predict'] = torch.from_numpy(data['rf_signal2_predict'])
        data['rf_signal1_predict'] = torch.stack([data['rf_signal1_predict'].real, data['rf_signal1_predict'].imag], dim=0).to(torch.float32)
        data['rf_signal2_predict'] = torch.stack([data['rf_signal2_predict'].real, data['rf_signal2_predict'].imag], dim=0).to(torch.float32)
        data['rfsignal1'] = torch.from_numpy(data['rfsignal1'])
        data['rfsignal2'] = torch.from_numpy(data['rfsignal2'])
        data['rfsignal1'] = torch.stack([data['rfsignal1'].real, data['rfsignal1'].imag], dim=0).to(torch.float32)
        data['rfsignal2'] = torch.stack([data['rfsignal2'].real, data['rfsignal2'].imag], dim=0).to(torch.float32)
        one_data = {'mix': mix, 'clean1': clean1, 'clean2': clean2, 'bits1': bits1, 'bits2': bits2, 'labels1': labels1, 'labels2': labels2, 'data':data}
        return one_data

def align_phase(f_off1, f_off2, phi1, phi2, clean1, clean2):
    f1 = f_off1
    f2 = f_off2
    phi1 = phi1
    phi2 = phi2

    n = np.arange(len(clean1))
    t = n / fs

    # 理想补偿 CFO + 相位
    clean1_c = clean1 * np.exp(-1j * (2 * np.pi * float(f1) * t + float(phi1)))
    clean2_c = clean2 * np.exp(-1j * (2 * np.pi * float(f2) * t + float(phi2)))
    return clean1_c, clean2_c

if __name__ == '__main__':
    dataset = PCMAModDataset_Decoding_Generate(shards_num=1)
    # for k in range(1, 99999, 5000):
    print(dataset[0])