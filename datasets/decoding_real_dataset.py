import torch
from torch.utils.data import Dataset, Subset
import glob
import os
import tqdm
# from torch.utils.data import 
from .base_dataset import PCMAModDataset
import random

class PCMAModDataset_Decoding_Generate_Real(PCMAModDataset):
    def __init__(self, transform=None, base=(
        # mixedmods_train_target_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.30]_snrU[15,15]_N250000_mod8P_varsnr_ampr_phi1phi2_delay0T_
            "/nas/datasets/yixin/PCMA/8PSK/target/"
            "mixedmods_train_target_rand_freqU[0,200]_"
            "phi1U[0.0000,6.2832]_"
            "phi2U[0.0000,6.2832]_"
            "ampU[0.30,0.30]_"
            "snrU[15,15]_"
            "N250000_mod8P_varsnr_ampr_phi1phi2_delay0T_c64_"
        ), shards_list=None, pattern=None, file_list=[]):
        super().__init__(file_list=file_list, transform=transform, base=base, shards_list=shards_list, pattern=pattern)

    def update(self, one_sample, idx):
        # 更新第idx个样本
        # print(one_sample.shape)
        clean1 = one_sample[0].cpu().numpy() + 1j * one_sample[1].cpu().numpy()
        clean2 = one_sample[2].cpu().numpy() + 1j * one_sample[3].cpu().numpy()
        # shard_idx = idx // self.m
        shard_id = int(torch.searchsorted(self.cum_sizes, idx, right=True))
        shard_start = 0 if shard_id == 0 else int(self.cum_sizes[shard_id - 1])
        local_idx = idx - shard_start

        # self._load_shard(shard_id)
        # data = self.current_data[local_idx]
        self.datas[shard_id][local_idx]['rf_signal1_predict'] = clean1
        self.datas[shard_id][local_idx]['rf_signal2_predict'] = clean2
        
        # self.current_data[local_idx] = data

    def __getitem__(self, idx):
        # 找到 idx 属于哪个 shard
        shard_id = int(torch.searchsorted(self.cum_sizes, idx, right=True))
        shard_start = 0 if shard_id == 0 else int(self.cum_sizes[shard_id - 1])
        local_idx = idx - shard_start

        self._load_shard(shard_id)
        data = self.current_data[local_idx]
        return data

if __name__ == '__main__':
    dataset = PCMAModDataset_Decoding_Generate_Real(shards_num=10)
    for k in range(1, 99999, 5000):
        print(dataset[k])