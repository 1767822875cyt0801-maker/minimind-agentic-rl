定义了不同训练阶段，并把不同训练阶段的数据转换成Pytorch训练时可以直接用的样本格式

定义了五类数据集，对应五种训练/对齐场景
(1) PretrainDataset 普通语言模型预训练
(2) SFTDataset 监督微调
(3) DPODataset 偏好学习
(4) RLAIFDataset 强化学习阶段的prompt构造
(5) AgentRLDataset 带工具的 agent 训练/评估数据准备


- **Pretrain**：`str -> token ids -> labels`
- **SFT**：`conversations -> prompt -> token ids -> 只监督 assistant`
- **DPO**：`chosen/rejected -> 两套 token ids -> 回答区间 mask -> x/y/mask`
- **RLAIF**：`conversations -> prompt`
- **AgentRL**：`conversations -> messages/tools + gt`



#### pre_processing_chat(conversations, add_system_ratio=0.2)

**功能**：对普通聊天数据，做轻量化system prompt；对tool数据，不动()
**输入**：`conversations: list[dict]`
```
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

**流程**：
- 检查是否有tools
- 如果有tool use 数据，直接返回
- 如果第一条不是`system`，按概率插入一条随机 system prompt
**输出**：`conversions：list[dict]`


#### post_processing_chat(prompt_content, empty_think_ratio=0.2)

**功能**：对已经渲染好的 prompt 字符串做后处理，不直接处理 token ids
**输入**：`prompt_content: str`
**流程**：
- 检查是否包含空的`<think>\n\n</think>\n\n`
- 以一定的概率删掉它
**输出**：`prompt_content：str`



#### PretrainDataset

纯文本 → token ids → 固定长度序列 → pad 位置 mask 掉

**训练场景**：普通预训练
**初始数据类型**：`sample：dict
```
{"text": "..."}
```
**输出数据类型**：`input_ids: torch.LongTensor`，`labels；torch.LongTensor`

**token处理变化流程**：纯文本 → token ids → 固定长度序列 → pad 位置 mask 掉

##### **`__getitem__` 的完整token流程
输入：`sample：dict`    `sample["text"]: str`
(1) 纯文本 → token ids
```
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids
```
数据变化：`str->list[int]`

**(2) 手动加 BOS / EOS**
```
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
```
数据变化：`list[int]->list[int]`  (多了BOS和EOS的token id)

**(3) padding**
```
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
```
数据变化：`list[int]->list[int]` （长度变为max_length）

**(4) 转tensor**
```
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
```
数据变化： input_ids  `list[int]->torch.LongTensor`

**(5)构造 labels**
```
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```
数据变化：labels  `labels: torch.LongTensor`
其中，普通token位置是真实token_id， pad位置是-100



#### SFTDataset

多轮对话 → 模板字符串 → 完整 token 序列 → 只给 assistant 回答打标签
把一条“多轮聊天 JSON 数据”变成语言模型训练时需要的 `(input_ids, labels)`
其中：
- input_ids：模型真正看到的输入token序列
- labels：训练目标
	- assistant回复部分：`labels[i] = input_ids[i]`
	- `user/system` 等非回答部分：`labels[i] = -100`
	`-100` 的含义是：
		- 在 PyTorch 的交叉熵里，`-100` 会被忽略
		- 也就是这些 token **不参与 loss**

**训练场景**：监督微调
**初始数据类型**：`sample:dict`   `sample["conversations"]:list[dict]`
```
{
    "conversations": [
        {"role": "...", "content": "...", ...},
        ...
    ]
}
```

##### create_chat_prompt(conversations)
**输入类型**：`list[dict]`
**功能**：
- 解析tools
- 解析tool_calls
- 调用tokenizer.apply_chat_template(...)
**输出类型**：`prompt: str`

##### generate_labels(input_ids)
**输入类型**：`input_ids：list[int]`
**功能**：
- 先扫描全-100的labels
- 扫描 `bos_id = tokenizer(f'{bos_token}assistant\n', ...)`
- 扫描 `eos_id = tokenizer(f'{eos_token}\n', ...)`
- 只把 assistant 正文和 eos 位置设成真实 token id
**输出类型**：`labels:list[int]`

##### `__getitem__` 的完整token流程
**输入**：`sample['conversation']`

**(1) 结构预处理**
`conversations = pre_processing_chat(sample['conversations'])`
- `list[dict] -> list[dict]`

**(2) 模板渲染**
`prompt = self.create_chat_prompt(conversations)`
- `list[dict]->str`

**(3) 字符串后处理**
`prompt = post_processing_chat(prompt)`
- `str -> str`

**(4) tokenizer**
`input_ids = self.tokenizer(prompt).input_ids[:self.max_length]`
- `str -> list[int]`

**(5) padding**
`input_ids += [pad_id] * (self.max_length - len(input_ids))`
- `list[int] -> list[int]（固定长度）`

**(6) 生成labels**
`labels = self.generate_labels(input_ids)`
- `list[int] ->list[int]`

**(7) 转tensor**
`torch.tensor(input_ids), torch.tensor(labels)`
- `list[int] -> torch.LongTensor`


#### DPODataset

两条完整对话 → 两套 token 序列 → 提取回答段 mask → 再切成 x/y/mask 供 DPO 比较概率

**训练场景**：偏好学习/DPO
**初始数据类型**：`chosen:list[dict]`     `rejected:list[int]`
```
{
    "chosen": [...],
    "rejected": [...]
}
```
chosen/rejected是一个list，里面有很多{role:    , content:     }


##### generate_loss_mask(input_ids)
**输入类型**：`input_ids: list[int]`
**功能**：
- 扫描assistant段开始边界`bos_id`
- 扫描结束边界`eos_id`
- 只给assistant正文和eos标一，其他位置标0



##### `__getitem__` 的 token 流程
**输入**：`chosen:list[int]`

**(1) 模板渲染**：
```
chosen_prompt = self.tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False
)
```
- `list[dict]->str`

**(2) 字符串后处理**：
```
chosen_prompt = post_processing_chat(chosen_prompt)
```
- `str -> str`

**(3) tokenizer**
```
chosen_encoding = self.tokenizer(
    chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
)
```

**(4) 取出 token ids**
```
chosen_input_ids = chosen_encoding['input_ids']
```
- `list[int]`

**(5) 生成回答区域mask**
```
chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
```
- `list[int] -> list[int]`

**(6) 做自回归shift**
```
x_chosen = chosen_input_ids[:-1]
y_chosen = chosen_input_ids[1:]
mask_chosen = chosen_loss_mask[1:]
```
- `list[int] -> list[int]`

**(7) 转tensor**
```
list[int] -> list[int]
```
- `list[int] -> LongTensor`


---
#### RLAIFDataset

这里只构造 prompt，不在 dataset 里 tokenize

**训练场景**：RL from AI Feedback阶段
**初始数据类型**：`sample["conversations"]: list[dict]`
```
{
    "conversations": [...]
}
```

##### create_chat_prompt(conversations)
**输入数据类型**：`list[dict]`
**功能**：
- pre_processing_chat(conversations)
- 随机决定是否use_thinking
- 取conversations[:-1]
- 调apply_chat_template(..., open_thinking=..., add_generation_prompt=True)
**输出类型**：`str`

##### `__getitem__` 的处理流程
**输入**：`sample['conversations']   # list[dict]`

**(1) 构造prompt**
`prompt = self.create_chat_prompt(sample['conversations'])`

**(2) 返回**
```
{
    'prompt': prompt,
    'answer': ""
}
```

**输出类型**：
- `prompt:str`
- `answer:str (空字符串)`



---

#### AgentRLDataset


**完全不在 dataset 里做 tokenize。**  
它保留的是更高层的结构化 agent 输入。


**训练场景**：agent+tool use+RL/评估
**初始数据类型**：`conversations：list[dict]`      `gt:任意结构化目标`
```
{
    "conversations": [...],
    "gt": ...
}
```

##### parse_conversations(conversations)
**输入类型**：`list[dict]`
**功能**：
- 遍历所有消息
- 如果 system message中有tools，就把他解析成结构化对象
- 所有消息加入message
- 返回message[:-1]和tools
输出：`(messages, tools)`
- `message:list[dict]`
- `tools:dict/list/None`

##### __getitem__
**输入**：
```
sample['conversations']   # list[dict]
sample['gt']
```

**(1) 解析对话**
`messages, tools = self.parse_conversations(sample['conversations'])`

**(2) 返回**
```
{  
'messages': messages,  
'tools': tools,  
'gt': sample['gt']  
}
```


---



这里只构造 prompt，不在 dataset 里 tokenize。







