它和预训练的核心区别在于：

- **预训练数据**：大规模原始文本，目标是学通用语言统计规律。
- **SFT 数据**：指令-回答、对话、多轮聊天样本，目标是学“按指令说话”的行为。
- **预训练关注**：语言建模能力、知识压缩、基础表达。
- **SFT 关注**：对齐任务格式、学会回答风格、遵守 instruction/chat 模板。
变化主要在
1.数据构造方式
- 数据集从PretrainDataset变成了SFTDataset
2.初始化方式（从pretrain权重开始）
- full SFT默认--from_weight pretrain，预训练默认--from_weight none
3.超参数（学习率更小，训练更稳）
- 这份 full SFT 脚本把一组默认超参数改成了更符合微调的设置：
	- `batch_size`: 16，而预训练是 32
	- `learning_rate`: `1e-5`，而预训练是 `5e-4`
	- `accumulation_steps`: 1，而预训练是 8
	- `max_seq_len`: 768，而预训练是 340
	- `data_path`: `sft_t2t_mini.jsonl`，而预训练是 `pretrain_t2t_mini.jsonl`
	- `save_weight`: `full_sft`，而预训练是 `pretrain`
	- `wandb_project`: `MiniMind-Full-SFT`，而预训练是 `MiniMind-Pretrain`
学习率更小，序列更长，梯度累积变小，直接一步一更


---

典型的full sft标准流程：
- 载入**预训练好的基座模型**
- 载入 tokenizer
- 读取 instruction / chat 数据
- 用 chat template 或 prompt template 把样本变成 token 序列
- 构造 `input_ids` 和 `labels`
- 前向计算 loss
- 反向传播，更新**全部参数**
- 定期记录日志、保存 checkpoint、支持 resume

---
工程上，full SFT 代码一般也分四层：

- **模型层**：模型结构定义
- **数据层**：SFTDataset / chat template / label mask
- **训练工具层**：lr、日志、checkpoint、resume、distributed
- **训练主脚本**：把以上模块接起来

MiniMind 的 `train_full_sft.py` 就是很典型的“训练主脚本层”。  
它自己不定义模型结构，也不定义 SFT 数据加工细节；它只是导入：

- `MiniMindConfig`
- `SFTDataset`
- `init_model`
- `lm_checkpoint`
- `SkipBatchSampler`

然后把这些模块编排起来

---
