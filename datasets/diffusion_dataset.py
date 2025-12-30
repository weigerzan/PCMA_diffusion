import torch
from torch.utils.data import Dataset, Subset
import glob
import os
import tqdm
# from torch.utils.data import 
from .base_dataset import PCMAModDataset
import random

class PCMAModDataset_Diffusion(PCMAModDataset):
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
        # 'mixsignal'为混合数据，'rfsignal1'和'rfsignal2'为对应干净数据，都是torch.complex64
        mix = torch.from_numpy(data['mixsignal'])
        clean1 = torch.from_numpy(data['rfsignal1'])
        clean2 = torch.from_numpy(data['rfsignal2'])
        # 转换为3072 * 2
        mix = torch.stack([mix.real, mix.imag], dim=0).to(torch.float)  # [2, 3072]
        clean1 = torch.stack([clean1.real, clean1.imag], dim=0).to(torch.float)
        clean2 = torch.stack([clean2.real, clean2.imag], dim=0).to(torch.float)

        # TODO 归一化怎么做？暂定对mix做mean-var归一化
        mix_mean = mix.mean()
        mix_std = mix.std()
        mix = (mix - mix_mean) / mix_std
        # # 测试一下，这里试图将noise按照功率比分配给两路
        
        clean1 = (clean1 - mix_mean * 0.5) / mix_std
        clean2 = (clean2 - mix_mean * 0.5) / mix_std

        noise = mix - clean1 - clean2
        # print(noise.std())
        clean1_norm = clean1.norm()
        clean2_norm = clean2.norm()

        data = {'mix': mix, 'clean1': clean1, 'clean2': clean2}
        return data

    def save_shards(self, base_path):
        for shard_id in range(len(self.datas)):
            torch.save(self.datas[shard_id], os.sep.join([base_path, 'shard{}.pth'.format(shard_id+1)]))

def get_train_test_with_config(config):
    trainset = PCMAModDataset_Diffusion(file_list=config.data.train_files)
    testset = PCMAModDataset_Diffusion(file_list=config.data.test_files)
    return trainset, testset

def get_test_with_config(config):
    testset = PCMAModDataset_Diffusion(file_list=config.data.test_files)
    return testset

if __name__ == '__main__':
    dataset = PCMAModDataset_Diffusion(shards_num=10)
    for k in range(1, 99999, 5000):
        print(dataset[k])