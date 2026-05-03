训练阶段的“辅助工具箱”，统一管理训练脚本运行时反复用到的一些公共功能

典型的训练流程如下：

**(1) 初始化训练环境**
init_distributed_mode()  看是否是多卡训练
setup_seed(seed)  固定随机种子

**(2) 初始化模型和 tokenizer**
`model, tokenizer = init_model(...)`
- 读tokenizer
- 创建MiniMindForCausalLM
- 如果有预训练参数/微调权重，就加载进来
- 打印参数量

**(3)如果是断点续训，加载 checkpoint**
`ckp = lm_checkpoint(..., model=None)`

**(4)构造 dataloader，必要时跳过前面已训练 batch**
如果之前已经训练过一些 step，那么恢复训练时：
- 可以用 `SkipBatchSampler`
- 跳过已经处理过的 batch

**(5)训练过程中动态计算学习率**

**(6)只在主进程打印日志**

**(7)定期保存 checkpoint**

**(8)如果是 RL/RLAIF 训练，还要调用 Reward Model 打分**





### 详细代码解释
1.**文件开头与路径处理**
```
"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
- `__file__`表示当前文件路径
- `os.path.dirname(__file__)`取当前文件所在目录
- `os.path.join(..., '..')`拼上上一级目录
- `os.path.abspath(...)`变成绝对路径
- `sys.path.append(...)`把这个目录加入Python模块搜索路径

---


2.**基础依赖导入**
```
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM
```

---


3.**get_model_params**
```
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')
```

`n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))`
如果config有n_routed_expert，就取它；否则尝试取num_experts；如果也没有，就取0

该函数不是只看“模型总共多少参数”(total)，还看“推理/训练时一次真正用了多少参数”(active= base+MOE)

---


4.**is_main_process和Logger**
```
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
```
在单卡/非 DDP 时：`dist.is_initialized()` 是 `False`
在DDP时，只有 `rank == 0` 返回 `True`
即单卡时，默认当前就是主进程；多卡时，只有 0 号进程是主进程

```
def Logger(content):
    if is_main_process():
        print(content)
```

只允许主进程打印日志

---

5.**get_lr**
```
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```
余弦退火学习率
目标：让学习率随时间按余弦曲线平滑衰减
$$\eta_{t} = \eta_{min} +\frac{1}{2}(\eta_{max}-\eta_{min})(1+cos(\frac{T_t}{T_{max}}\pi))$$
- $\eta_t$：第t步的学习率
- $\eta_{max}$：初始最大学习率
- $\eta_{min}$：最终最小学习率
- $T_{t}$：当前周期的步数
- $T_{max}$：单个周期的总步数

==改进：带warmup的余弦退火==

6.**init_distributed_mode**
```
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```
初始化分布式训练环境
`os.environ.get("RANK", -1)`
`os.environ` 是 Python 里访问**环境变量**的接口。在DDP启动时，通常环境变量里会有：RANK(全局进程编号)，LOCAL_RANK(当前机器上的本地进程编号)，WORLD_SIZE（总进程数）
去环境变量里找 `"RANK"`， 如果存在，返回它的值，如果不存在返回默认值-1.如果没有RANK，说明当前不是用torchrun/DDP 启动的,那就按普通单进程模式处理

`dist.init_process_group(backend="nccl")`
初始化Pytorch分布式进程组，把当前进程加入到一个“分布式通信组”里，使多个训练进程能够互相通信
`backend` 指分布式通信后端 
- `"nccl"`：主要用于 GPU 训练，性能最好
- `"gloo"`：CPU 或兼容性更广
- `"mpi"`：某些 HPC 环境使用

`local_rank = int(os.environ["LOCAL_RANK"])`
获取当前进程绑定的本地 GPU 编号

`torch.cuda.set_device(local_rank)`
把当前进程绑定到对应 GPU

7.**setup_seed**
```
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

固定随机种子，尽可能保证实验可复现

8.**lm_checkpoint**
负责保存/加载checkpoint
- `model is not None` → 保存模式
- `model is None` → 加载模式

- **Pytorch保存和加载通用checkpoint**
保存
```python
# 保存 checkpoint 文件
ckpt_path = "model_checkpoint.pth"
torch.save({
    'epoch': epoch,  # 保存当前的epoch
    'model_state_dict': model.state_dict(),  # 保存模型的state_dict
    'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的state_dict
    'loss': loss,  # 保存最后一次的损失值
}, ckpt_path)
```
加载
```python
text
# 加载 checkpoint 文件
checkpoint = torch.load(ckpt_path)

# 恢复模型的权重
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器的状态
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复训练轮次和损失值
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

```
ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
```
构造纯模型权重文件路径，这个文件后面只会存储state_dict，也就是模型参数，不包含 optimizer、epoch 等训练状态。只关心模型参数。适合部署和推理

```
resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
```
构造恢复训练用的完整状态文件路径。需要包含完整训练状态。适合断点续训



**保存逻辑**：
1.`raw_model = model.module if isinstance(model, DistributedDataParallel) else model`
因为在DDP训练中，真实模型通常会被包成`model = DistributedDataParallel(raw_model, ...)`

2.`raw_model = getattr(raw_model, '_orig_mod', raw_model)`
某些情况下，模型可能还被别的机制包装过，比如：
- `torch.compile`
- 某些自定义 wrapper
- 其他优化框架
有时原始模型对象会被挂在 `_orig_mod` 属性里。

3.`state_dict = raw_model.state_dict()`
`state_dict()` 会返回一个字典，里面保存了模型所有参数和 buffer

4.`state_dict = {k: v.half().cpu() for k, v in state_dict.items()}`
把参数转成 half 并搬到 CPU

5.`ckp_tmp = ckp_path + '.tmp'`

6.`torch.save(state_dict, ckp_tmp)`

7.`os.replace(ckp_tmp, ckp_path)`
用新文件替换旧文件，如果目标文件已存在，也直接覆盖
原子写入思路：先写临时文件，写成功后，再replace成正式文件




---

8.**init_model**
初始化训练/推理所需的 tokenizer 和 MiniMind 模型，并按需要加载已有权重，最后把模型放到指定设备上

和下面这些函数/class有关：
- 和 `MiniMindForCausalLM` 有直接关系。`MiniMindForCausalLM` 负责“模型长什么样”，而`init_model` 负责“什么时候把它创建出来，并决定要不要加载已有参数”
- 和 `lm_checkpoint` 有间接关系，`init_model` 负责“初始化模型并加载基础权重”，`lm_checkpoint` 负责“保存/恢复训练状态”


**函数定义**：
`def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):`
- lm_config 模型配置对象
- from_weight 初始化时要从哪一类权重开始加载
- tokenizer_path    tokenizer 从哪个路径读取
- save_dir  模型权重文件所在目录，同时也可能是后续训练输出目录
- device 模型放到哪个设备

**加载 tokenizer**：
`tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)`
AutoTokenizer是transformers里的自动工厂类，它会自动读取后面tokenizer_path里的配置文件自动判断应该实例化哪种tokenizer，然后from_pretrained(...)构造对象

**创建模型对象**：
建模型
`model = MiniMindForCausalLM(lm_config)`
加载权重
```
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
```
`weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'`
按照项目约定的命名规则拼出待加载的权重文件路径
(在之前已经按照这个命名规则在该目录下存了权重文件，现在按照这个规则去读取)
- save_dir 权重所在目录
- from_weight 权重类别名
- lm_config.hidden_size  区分不同规模模型
- moe_suffix   是否为 MoE 的后缀
- .pth  Pytorch 常见权重文件后缀

`weights = torch.load(weight_path, map_location=device)`
从磁盘读取权重文件，并把其中的张量映射到指定设备

`model.load_state_dict(weights, strict=False)`
把权重载入模型

---
9.**SkipBatchSampler**
Sampler 负责产生单个样本的索引顺序，一个接着一个index进行标记和弹出
batch sampler 把一个个单样本索引，组装成一批批的索引列表
SkipBatchSampler 继承自Sample，直接输出一个batch列表。它拿到底层sample产生的单索引流，再自己在上面做"分批和跳批"

外部参数skip_batches 常见来源：
- checkpoint直接保存已经跑过多少batch
- 只保存了 `step`，再结合训练配置换算
- 通过全局 step 和 dataloader 长度反推




#### ==DataLoader、DataSet, Sampler之间的关系==
- **Sampler**: 定义样本按照什么顺序/规则被访问
- **Dataset** ： 定义“数据长什么样、怎么根据一个样本标识把它读出来
- **DataLoader**: 把前两者组织起来，把前两者串起来，并额外处理 batching、collate、多进程、pin memory最后产出一个 batch
最经典的 map-style 情况:
```
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```
`batch_sampler` 给一组索引 -> `dataset[i]` 逐个取样本 -> `collate_fn` 拼成 batch -> `DataLoader` yield 给训练循环

##### Dataset
通常实现：
- `__getitem__(idx)`：给一个索引，返回一个样本
- `__len__()`：告诉外界一共有多少个样本
例如图像分类里:
```
class MyDataset(Dataset):
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        image = ...
        label = ...
        return image, label
```
Dataset不关心epoch、shuffle和batch，只负责给他一个索引，它把对应样本进行返回

##### Sampler
`Sampler` 不读取数据本身。  
它只产出 **索引**，也就是告诉 `DataLoader`下一个去取的idx是多少
常见Sampler：
- `SequentialSampler`：顺序取 `0,1,2,3,...`
- `RandomSampler`：随机打乱索引
- `SubsetRandomSampler`：只在某个子集里随机采样
- `WeightedRandomSampler`：按权重抽样，常用于类别不平衡
- `DistributedSampler`：分布式训练时把数据切到不同进程/卡上


##### DataLoader
DataLoader 把 dataset 和 sampler 组合起来，形成一个可迭代的数据流
DataLoader 本身不决定“抽谁”，它要么使用自动构造的 sampler，要么使用你传入的 sampler
它除了调用 sampler 和 dataset，还负责：
- `batch_size`
- `drop_last`
- `batch_sampler`
- `collate_fn`
- `num_workers`
- `pin_memory`
- `prefetch_factor`
- `persistent_workers`
- `worker_init_fn` 等
最常用的
```
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
顺序或 shuffled 的 sampler 会依据 `shuffle` 自动构造；也可以通过 `sampler=` 显式传自定义采样器
shuffle=True表示“我希望随机顺序访问样本”，如果没有自己传sampler，Pytorch会自动构造随机采样器
sampler=...表示“不要你默认决定了，我自己规定索引顺序”
batch_sampler=...表示：不是“每次给一个索引”，而是“每次直接给一组索引”
`batch_sampler` 与 `batch_size`、`shuffle`、`sampler`、`drop_last` 互斥。因为你都已经直接告诉它“每一批取哪些索引了”，它就不需要再自己 batch
BatchSampler(batch_size=3)   `BatchSampler` 包装另一个 sampler，并产出 mini-batch 的索引列表
DataLoader 往往是先通过 sampler 拿到单个索引，再经由 batch_sampler 组合成索引批，最后才去 Dataset 里取样本。



---
10.**LMForRewardModel**
一个对奖励模型/评分模型做的轻量封装，用来给某个回答打分。该类把一个外部 reward model 包装起来，提供统一的 `get_score` 接口
输入上下文和回答，输出reward score


`def __init__(self, model_path, device="cuda", dtype=torch.float16):`
model_path 表示reward model的路径
dtype=torch.float16   指定加载模型时使用的张量精度，默认为half precision 因为 reward model 一般只做推理评分，不反向传播，用半精度通常更省显存，更快，足够做inference

`self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`
trust_remote_code=True 表示允许transformers 加载模型仓库里自定义的 Python 代码实现，因为很多reward model或者chat model仓库会自定义：对话模版、打分函数、特殊前处理逻辑、自定义模型类

reward model 打分的核心往往是：
- 给它“当前问题”
- 给它“候选回答”
- 它判断这个回答对这个问题好不好
- 如果有多轮对话，就把前面对话作为历史上下文压缩，最后一条作为当前 query


















