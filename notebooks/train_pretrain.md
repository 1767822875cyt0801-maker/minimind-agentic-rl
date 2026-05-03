对于decoder-only语言模型，预训练最核心的任务通常是：

给模型一段 token 序列，让它根据前面的 token 去预测下一个 token。

也就是做 **causal language modeling**。训练时一般会得到两类张量：

- `input_ids`：输入给模型的 token id 序列
- `labels`：监督信号，通常和 `input_ids` 错位对应，用来计算交叉熵损失

模型前向以后输出 logits，再由 logits 和 labels 计算 loss，然后反向传播，更新参数。


---

一个比较典型的预训练脚本，逻辑上通常是下面这条链：

**命令行参数 / 配置**  
→ **初始化设备、随机种子、分布式环境**  
→ **构建模型与 tokenizer**  
→ **构建 dataset / dataloader**  
→ **构建 optimizer / lr scheduler / scaler**  
→ **进入 epoch-step 训练循环**  
→ **每一步做 forward / loss / backward / optimizer step**  
→ **定期日志记录、保存 checkpoint、支持 resume**


----

这份 `train_pretrain.py` 的定位，不是“定义模型”，也不是“定义数据集”，而是**把模型、数据、优化器、分布式、混合精度、日志、保存/续训这些模块串成一个可跑的预训练主程序**。
模型来自 `MiniMindConfig`，数据来自 `PretrainDataset`，训练工具来自 `trainer_utils`


Dataset = `lm_dataset.py` 里的 `PretrainDataset`
`PretrainDataset`继承父类Pytorch中的Dataset，负责提取json/jsonl文件中的数据并转化成(input_ids, labels)

DataLoader = `train_pretrain.py` 里构造的 `loader`，DataLoader的实例

而“Sampler”实际上分成了两层：
1. **底层样本顺序来源**：分布式时是 `DistributedSampler(train_ds)`，非分布式时是 `torch.randperm(len(train_ds)).tolist()` 这串随机索引。
2. **真正直接喂给 DataLoader 的批采样器**：`SkipBatchSampler(train_sampler or indices, args.batch_size, skip)`。也就是说，这里最终不是 `DataLoader(batch_size=..., shuffle=...)` 在决定 batch，而是 `batch_sampler` 在决定“每个 batch 包含哪些样本索引”。PyTorch 官方文档也明确说了：对 map-style dataset，`DataLoader` 在这种情况下大致等价于 `for indices in batch_sampler: yield collate_fn([dataset[i] for i in indices])`

len(dataset):样本总数
len(sampler):采样器要产生多少索引
len(dataloader):一个epoch有多少个batch







### train_epoch()
功能：执行一个epoch的训练过程，跑完一个epoch里面所有的step
```
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
```
- `epoch`：当前第几个 epoch
- `loader`：当前 epoch 对应的 DataLoader
- `iters`：这个 epoch 应该视作总共有多少 step
- `start_step`：如果是断点续训，从哪个 step 之后开始
- `wandb`：实验记录对象，可能是 `None`，也可能是 `swanlab as wandb` 的别名模块对象

```
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
```
说明`PretrainDataset + DataLoader` 每次会产出一对：(input_ids，labels)，标准语言模型训练格式



### if __name__ == "__main__"

1.**命令行参数**
```
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
```



2.**初始化环境和随机种子**
```
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```
args.device = f"cuda:{local_rank}"  让每个进程绑定到自己的GPU

3.**配置目录、模型参数、检查点**
```
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

4.**配wandb**
SwanLab 专注训练实验记录、可视化和对比的工具
W&B  以实验跟踪为核心、向调参、工件管理、样本分析和团队协作扩展的平台
```
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
```


5.**定义模型、数据、优化器**
```
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```
1.init_model(...)
它负责根据 `lm_config` 构建模型、根据 `from_weight` 决定是否加载预训练权重、把模型放到指定设备并返回tokenizer
2.PretrainDataset(...)
这个数据集对象应该负责读取 `jsonl`，对文本tokenize，构造训练样本，并返回(input_ids,labels)
3.DistributedSampler
如果是 DDP，就需要它来确保不同 rank 看到不同数据分片。  否则每张卡都训练同样的样本，就没有意义。
4.GradScaler
只在 `float16` 下启用，和前面的 AMP 设置配套
5.AdamW
这是优化器。这里没有复杂参数组划分，直接 `model.parameters()` 全量进入 AdamW

6.从ckp恢复状态
```
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
```
标准断点续训逻辑
这里是恢复现场，而不仅仅是模型参数，所以恢复的不只是模型权重，还包括optimizer状态、scaler状态以及epoch/step，还有AdamW的动量项，AMP的缩放状态


7.编译和分布式包装
训练加速层
```
if args.use_compile == 1:
    model = torch.compile(model)
    Logger('torch.compile enabled')
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```
如果打开use_compile，会用PyTorch 2 的图编译能力优化模型执行
`_ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}`某些buffer不参与同步或者特殊处理。这两个是 RoPE 预计算得到的 buffer。它们通常是确定性的常量缓存，不需要像参数一样参与梯度同步
`DistributedDataParallel()` DDP包装，把模型包装成DistributedDataParallel，这样backward时会自动同步梯度

8.开始训练
```
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
```
(1)按epoch循环
外层控制epoch数
(2)DDP sampler设置
DDP 下如果 sampler 不按 epoch 改随机种子，每个 epoch 的数据打乱顺序可能不变
(3)重新设种子并生成随机索引
如果不是 DDP，它就手动生成一个乱序索引列表
(4)准备索引 / 跳过步数
(5)创建 DataLoader
(6)调用 `train_epoch`






