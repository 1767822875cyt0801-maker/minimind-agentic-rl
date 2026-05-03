包含所有自回归文本生成函数的类，用作模型类的 mixin
给具体语言模型提供 `.generate()` 能力的一层通用生成框架。语言模型自己负责 `forward` 算 logits，`GenerationMixin` 负责把“输入准备 → cache 管理 → 解码策略 → 停止条件 → 输出组织”整套流程串起来。


### generate()


**输入参数**
1.真正的语言模型输入:
- decoder-only模型，inputs一般是inputs_ids
- encoder-decoder模型，`inputs` 可以是 `input_ids`、`input_values`、`input_features``pixel_values` 等编码器输入

2.生成控制参数
最常见的是 `generation_config` 和那些会覆盖它的 `kwargs`
- 生成多长
- greedy 还是 sample
- beam 数是多少
- 温度、top-k、top-p
- 是否返回分数、hidden states、attentions
- 是否返回结构化输出等

3.高级扩展输入


**输出**：
1.默认返回torch.LongTensor
通常直接返回生成后的 token 序列
对于decoder-only模型，返回的张量形状一般是
[batch_size * num_return_sequences, total_sequence_length]
其中 `total_sequence_length = prompt_len + generated_len`

2.返回 `ModelOutput`


**generate()流程:**
- **准备输入**：把 `inputs`、`input_ids`、`inputs_embeds`、encoder 输入等整理成统一形式。源码里对应 `_prepare_model_inputs()`。
- **准备生成态输入**：比如 position ids、attention mask、cache 相关内容。源码里对应 `prepare_inputs_for_generation()`、`_prepare_attention_mask_for_generation()` 等。
- **进入解码循环**：每一步调用模型 `forward` 得到下一步 logits。这个“循环调 forward”的公共逻辑由 `GenerationMixin` 负责。
- **依据策略选 token**：greedy / sample / beam / assisted。
- **更新状态**：把新 token 接到序列后面，更新 `past_key_values`、`attention_mask`、`position_ids` 等。源码里对应 `_update_model_kwargs_for_generation()`。
- **判断是否停止**：达到 `max_new_tokens`、`max_length`、或者生成到 EOS。




##### 以decoder-only为例
`generate()` 的本质就是两段：
- **Prefill**：把整段 prompt 一次性喂进模型，建立每一层的 KV cache
- **Decode loop**：之后每轮只喂“还没处理过的新 token”，通常就是 1 个 token；模型读取旧 cache，补上当前 token 的 K/V，再预测下一个 token。

把“生成参数解析”和“循环控制”写在 `generate()` 里。`forward()` 只负责一次前向；`generate()` 负责反复调用 `forward()`
- 调用model.generate()
	- 合并 `generation_config` 和你传入的 `kwargs`
- `_prepare_model_inputs()`：先把入口输入整理好
	- 生成开始前，源码会先做一次输入整理。`_prepare_model_inputs()` 的职责是抽取模型真正要用的输入，处理 `inputs` / `input_ids` / `inputs_embeds` 的关系
- 准备 `attention_mask` 和 `position_ids`
	- 准备 mask 和位置
- 第一次前向：这就是 **prefill**
	- `generate()` 会调用 `prepare_inputs_for_generation()`。这个函数的文档写得很明确：它会为生成准备模型输入，包括选择正确的输入键、在缺失时创建 `position_ids`、**切分 inputs**、必要时处理 attention mask/cache，并把其余参数继续传给模型的 `forward()`
- Prefill 的结果：拿到第一份 KV cache
- `_update_model_kwargs_for_generation()`：把这轮结果接回下一轮
	- 根据刚生成出的新 token，更新 `attention_mask`、`position_ids`、`token_type_ids`，并把输出中的 cache 写回 `model_kwargs`
- 第二次及以后前向：这就是 **decode**
	- 有 `past_key_values` 时，应只输入 **未处理过的** `input_ids`，形状是 `(batch_size, unprocessed_length)`，而不是整个 `(batch_size, sequence_length)`。对标准一步一步生成来说，这个 `unprocessed_length` 通常就是 `1`





