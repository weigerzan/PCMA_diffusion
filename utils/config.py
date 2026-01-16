from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    type: str
    predict_target: str
    num_diffusion_timesteps: int

@dataclass
class DataSplitConfig:
    paths: Optional[List[str]] = None
    base: Optional[str] = None
    shard_list: Optional[List[int]] = None
    pattern: Optional[str] = None

@dataclass
class DataConfig:
    signal_len: int
    modulation: str
    train: DataSplitConfig
    test: DataSplitConfig
    decoding: DataSplitConfig

@dataclass
class TrainingConfig:

    train_batch_size: int
    test_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int

    learning_rate: float
    lr_warmup_steps: int

    

    mixed_precision: str
    seed: int

    save_data_epochs: int
    save_model_epochs: int
    output_dir: str
    overwrite_output_dir: bool

    push_to_hub: bool
    hub_model_id: str
    hub_private_repo: bool

    pretrained: Optional[str] = None

@dataclass
class SamplingConfig:
    num_inference_steps: int
    eta: float
    output_dir: str

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    sampling: SamplingConfig

def build_file_list(split_cfg: DataSplitConfig):
    """Unify paths, or base+shard_list+pattern, into a clean list of file paths."""
    # Option A — user provides explicit paths
    if split_cfg.paths is not None and len(split_cfg.paths) > 0:
        return split_cfg.paths

    # Option B — base + shard_list + pattern
    if split_cfg.base is None or split_cfg.pattern is None:
        return []
    
    base = Path(split_cfg.base)
    pattern = split_cfg.pattern
    
    # 如果shard_list不为空，直接使用
    if split_cfg.shard_list is not None and len(split_cfg.shard_list) > 0:
        return [
            str(base / pattern.format(idx=i))
            for i in split_cfg.shard_list
        ]
    
    # 如果shard_list为空，尝试从文件系统中查找匹配的文件
    if not base.exists():
        return []
    
    # 将pattern转换为glob模式：{idx:02d} -> *, {idx:01d} -> *
    import re
    pattern_glob = pattern
    pattern_glob = re.sub(r'\{idx:\d+d\}', '*', pattern_glob)
    
    # 查找匹配的文件
    matched_files = sorted(base.glob(pattern_glob))
    if matched_files:
        return [str(f) for f in matched_files]
    
    return []


def load_config(path: str) -> Config:
    """Load the YAML and build a unified config object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # print(**raw["training"])
    # Build dataclasses
    cfg = Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(
            signal_len=raw["data"]["signal_len"],
            modulation=raw["data"]["modulation"],
            train=DataSplitConfig(**raw["data"]["train"]),
            test=DataSplitConfig(**raw["data"]["test"]),
            decoding=DataSplitConfig(**raw["data"]["test"])
        ),
        training=TrainingConfig(**raw["training"]),
        sampling=SamplingConfig(**raw["sampling"]),
    )

    # Add actual train/test file lists
    cfg.data.train_files = build_file_list(cfg.data.train)
    cfg.data.test_files = build_file_list(cfg.data.test)

    # 设置decoding配置（用于推理后的解调）
    cfg.data.decoding.paths = []
    cfg.data.decoding.base = raw['sampling']['output_dir']
    cfg.data.decoding.pattern = 'shard{idx:01d}.pth'
    
    # 确定decoding的shard_list：基于test数据的文件数量
    # 支持两种方式：paths 或 base+shard_list+pattern
    test_config = raw['data']['test']
    if 'paths' in test_config and test_config['paths'] is not None and len(test_config['paths']) > 0:
        # 方式1：使用paths列表
        num_test_files = len(test_config['paths'])
    elif 'shard_list' in test_config and test_config['shard_list'] is not None and len(test_config['shard_list']) > 0:
        # 方式2：使用shard_list
        num_test_files = len(test_config['shard_list'])
    else:
        # 如果都没有，尝试从实际文件列表中获取
        num_test_files = len(cfg.data.test_files)
    
    # 如果仍然无法确定，默认使用1
    if num_test_files == 0:
        num_test_files = 1
    
    cfg.data.decoding.shard_list = list(range(1, num_test_files + 1))
    cfg.data.decoding_files = build_file_list(cfg.data.decoding)
    cfg.training.learning_rate = float(cfg.training.learning_rate)

    return cfg