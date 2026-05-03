MiniMind 的 LoRA 微调训练入口脚本，完成“加载底座模型 → 注入 LoRA → 冻结原参数 → 构造 SFT 数据 → 训练/续训 → 保存 LoRA 权重与 checkpoint”
在 MiniMind 基础模型上做参数高效微调（PEFT），只训练 LoRA 参数，不改动原模型大部分参数。

调用链：
```
命令行启动脚本
   ↓
argparse 解析参数
   ↓
初始化分布式 / 随机种子 / 混合精度
   ↓
构造 MiniMindConfig
   ↓
init_model(...) 加载模型和 tokenizer
   ↓
apply_lora(model) 注入 LoRA
   ↓
冻结非 LoRA 参数
   ↓
SFTDataset(...) 构造数据集
   ↓
DataLoader + Sampler 组织 batch
   ↓
for epoch:
    train_epoch(...)
        ↓
        model(input_ids, labels=labels)
        ↓
        backward / optimizer.step / scaler.update
        ↓
        save_lora + lm_checkpoint
```


### 模块功能
1.**环境与依赖导入**
```
import os

import sys

  

__package__ = "trainer"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

  

import argparse

import time

import warnings

import torch

import torch.distributed as dist

from contextlib import nullcontext

from torch import optim, nn

from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader, DistributedSampler

from models.model_minimind import MiniMindConfig

from dataset.lm_dataset import SFTDataset

from models.model_lora import save_lora, apply_lora

from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

  

warnings.filterwarnings('ignore')
```

2.**`train_epoch` 训练核心函数**
```

```
