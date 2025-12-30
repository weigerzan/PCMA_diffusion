import torch
from torch.utils.data import Dataset, Subset
import glob
import os
import tqdm
# from torch.utils.data import 


class PCMAModDataset(Dataset):
    def __init__(self, file_list=[], transform=None, base = (
        # mixedmods_train_target_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.30]_snrU[15,15]_N250000_mod8P_varsnr_ampr_phi1phi2_delay0T_
            "/nas/datasets/yixin/PCMA/8PSK/target/"
            "mixedmods_train_target_rand_freqU[0,200]_"
            "phi1U[0.0000,6.2832]_"
            "phi2U[0.0000,6.2832]_"
            "ampU[0.30,0.30]_"
            "snrU[15,15]_"
            "N250000_mod8P_varsnr_ampr_phi1phi2_delay0T_c64_"
        ), shards_list=None, pattern=None):
        """
        root_pattern: 
            "/nas/datasets/yixin/PCMA/8PSK/mixedmods_train_robust_..._shard*.pth"
        """

        # 要求要么提供file_list，要么提供base和shards_list
        if  file_list is None or len(file_list) == 0:
            assert (base is not None) and (shards_list is not None), "Either file_list or base path and shards_list must be provided."

        if  file_list is None or len(file_list) == 0:
            print('Preparing dataset from base path:', base)
            BASE = base
            self.files = [
                f"{BASE}/{pattern.format(idx=i)}"
                for i in shards_list
            ]
        else:
            self.files = file_list
        assert len(self.files) > 0, "No shard files found."

        self.transform = transform

        # 记录每个 shard 的样本数
        self.shard_sizes = []
        self.datas = []
        print('Prepare dataset')
        for f in tqdm.tqdm(self.files):
            data = torch.load(f, map_location="cpu", weights_only=False)
            self.shard_sizes.append(self._get_length(data))
            self.datas.append(data)
            del data
            

        # 累积索引，用于全局 index → shard
        self.cum_sizes = torch.cumsum(
            torch.tensor(self.shard_sizes), dim=0
        )

        self.current_shard_id = None
        self.current_data = None

    def _get_length(self, data):
        return len(data)

    def _load_shard(self, shard_id):
        self.current_data = self.datas[shard_id]

    def __len__(self):
        return int(self.cum_sizes[-1])

    def __getitem__(self, idx):
        # 找到 idx 属于哪个 shard
        shard_id = int(torch.searchsorted(self.cum_sizes, idx, right=True))
        shard_start = 0 if shard_id == 0 else int(self.cum_sizes[shard_id - 1])
        local_idx = idx - shard_start

        self._load_shard(shard_id)
        data = self.current_data[local_idx]

        return data


def get_train_test_100k():
    dataset = PCMAModDataset(shards_num=10)
    split = torch.load("/home/zhangjiawei/scripts/PCMA_diffusion/datasets/pcma_split_9to1_seed42.pth")
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
    print("train length:{}; test length:{}".format(len(train_idx), len(test_idx)))  # 90000 10000
    train_set = Subset(dataset, train_idx)
    test_set  = Subset(dataset, test_idx)
    return train_set, test_set


def get_train_test_100k_real():
    dataset = PCMAModDataset(shards_num=10, base=('/home/zhangjiawei/scripts/PCMA_diffusion/real_data/8psk/train/'))
    split = torch.load("/home/zhangjiawei/scripts/PCMA_diffusion/datasets/pcma_split_9to1_seed42.pth")
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
    print("train length:{}; test length:{}".format(len(train_idx), len(test_idx)))  # 90000 10000
    train_set = Subset(dataset, train_idx)
    test_set  = Subset(dataset, test_idx)
    return train_set, test_set


if __name__ == '__main__':
    dataset = PCMAModDataset(shards_num=10)
    for k in range(1, 99999, 5000):
        print(dataset[k])