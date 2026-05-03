知识蒸馏训练脚本
让一个学生模型同时学习两件事：
- 学真实标签labels(即普通CE监督)
- 学教师模型输出的soft target（即KL蒸馏损失）
总目标：把 teacher 的分布知识压缩到 student 里

复用了train_pretrain.py / full_sft.py的训练骨架
最大区别在于：

 `train_pretrain.py`
核心损失通常是 **next-token CE loss*

 `full_sft.py`
核心也是 **CE loss**，但数据是聊天监督格式，只对 assistant 部分算 loss

 `train_distillation.py`
核心损失变成：loss=α⋅CE+(1−α)⋅KL

### SFT 阶段的蒸馏 / 压缩
**蒸馏**：
- 有一个 **老师模型（teacher）**，通常更大、更强
- 有一个 **学生模型（student）**，通常更小、更便宜
- 训练时不只是让学生拟合“标准答案”，还要让学生去模仿老师的输出分布、行为风格、推理格式或中间表示
**压缩**：
压缩是更大的概念，包含但不限于蒸馏。常见的压缩手段有：
- 知识蒸馏
- 剪枝（pruning）
- 量化（quantization）
- 低秩分解（low-rank factorization）
- 参数共享
- 小模型结构设计
- 稀疏化

**SFT 阶段的蒸馏/压缩**：
- 在指令微调数据上做蒸馏
- 或者在 SFT 完成后，对这个 SFT 模型进一步做结构压缩
- 目标是保留 **指令理解、问答格式、聊天风格、任务完成能力**

**蒸馏到底在蒸什么？**

**1.答案本身**
- 输入给老师模型一个 instruction
- 老师生成一个高质量回答
- 学生拿这个回答做监督训练
可以看作response distillation，形式上还是普通SFT

**2.老师的 token 概率分布**
学生不只是学最终选中的 token，而是去学这个 **完整分布**，这叫 **logit distillation / soft target distillation**
常见损失是KL散度：
$$\mathcal{L}_{KD} = KL(p_{T}^\tau || p_{S}^\tau)$$
其中 τ 是 temperature，用来把概率分布“变软”

**3.中间层表示**
不仅输出层像老师，中间层也尽量像老师

**4.行为风格**



```
res = model(input_ids)
```
这里的model调用的是model_minimid中的MiniMindForCausalLM()，返回的是
```
return MoeCausalLMOutputWithPast(

            loss=loss,

            aux_loss=aux_loss,

            logits=logits,

            past_key_values=presents,

            hidden_states=hidden_states,

        )
```
res不是单个张量，而是一个打包好的返回对象，里面放了多个字段
- `res.loss`
- `res.aux_loss`
- `res.logits`
- `res.past_key_values`
- `res.hidden_states`
其中`res.logits`的维度是$[B,T,V]$
对于一整个长度为$V$的向量，表示：对于第 `i` 个样本的第 `t` 个位置，模型认为下一个 token 是词表中每个 token 的分数分别是多少


**计算loss的流程**：

**1.构造监督有效区域(masking)**
```
loss_mask = (labels[..., 1:] != -100).float()
```
构造一个 supervision mask，只保留真正应该参与损失计算的位置

**2.学生logits对齐(causal LM alignment)**
```
res = model(input_ids)
student_logits = res.logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

**3.真实标签损失（ground-truth CE）**
```
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
```
计算基于真实标签的 token-level cross entropy

**4.MoE 辅助损失融合（optional auxiliary regularization）**
```
if lm_config_student.use_moe:
    ce_loss = ce_loss_raw + res.aux_loss
else:
    ce_loss = ce_loss_raw
```

**5.蒸馏损失（teacher-student KL）**
```
teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
teacher_logits = teacher_logits[..., :vocab_size_student]

distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)





def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

在有效 token 上，对 student 和 teacher 的输出分布做 logits-level distillation

**维度变化：**

先对齐：
- `teacher_logits`: `[B, T-1, V_t]`-> ``[B, T-1, V_s]``
- `student_logits：[B, T-1, V_s]`

展平后：
-  `student_logits.view(-1, V_s)` → `[B*(T-1), V_s]`
- `teacher_logits.view(-1, V_s)` → `[B*(T-1), V_s]`

再按 mask 选有效 token：

- → `[N_valid, V_s]`

蒸馏函数内部：

- teacher logits → soft targets
- student logits → log probs
- KL → scalar

**6.总损失融合+梯度累积**
```
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```
把监督损失和蒸馏损失按比例融合，再适配梯度累积

==Notice==：view(...)并不能改掉张量本身，只是生成了一个临时视图，张量本身的维度并没有改变
例如：
```
F.cross_entropy(student_logits.view(-1, V), ...)
```
含义是：
- 先临时得到一个二维 view
- 把这个 view 传进函数
- 函数用完以后，这个临时 view 也就完成使命了
- `student_logits` 自己还是原来的三维张量

如果写成：
```
student_logits = student_logits.view(-1, V)
```
意味着：
- 重新把变量 `student_logits` 绑定到一个二维张量上
- 后面这个变量就不再是 `[B, T-1, V]` 了，而是 `[B*(T-1), V]`

==Notice==：教师模型不用从ckp恢复状态和编译和分布式包装？

**因为 teacher 在这个脚本里只是“前向提供软标签”，不是训练主体**，所以它不需要恢复 optimizer/scaler 这类训练状态；至于 compile 和 DDP，不是“不能”，而是这份实现里作者刻意只把这些优化放在 student 这边。
这份脚本里，**teacher 被定义成“固定的推理参考模型”，不是训练对象”**
1.为什么不用从 ckp 恢复“训练状态”
- 分清两种状态：
	- 模型权重参数
		- teacher 其实是会加载权重的，只是不是从“续训 checkpoint”恢复，而是通过`from_teacher_weight` 加载一个固定教师权重,student也是这样初始化的
	- 训练过程状态
		- 真正从 `ckp_data` 恢复的是：
			-  `model.load_state_dict(...)`
			- `optimizer.load_state_dict(...)`
			- `scaler.load_state_dict(...)`
			- `start_epoch`
			- `start_step`
		- teacher 不需要恢复这些训练状态，因为它根本不训练

2.为什么 teacher 不做 `torch.compile`
这不是“理论上不能”，而是**这份实现里没必要优先做**。
`torch.compile` 是 PyTorch 2.x 引入的即时（JIT）编译器，它通过将模型的计算图捕获并编译为优化的内核，显著提升大模型（LLM）的训练和推理效率。
student是整个训练主路径上最重、最核心的对象。teacher 只做前向参考，所以作者把 compile 资源优先给 student，是很自然的取舍

3.为什么 teacher 不做分布式包装（DDP）
DDP 的核心作用是多进程训练时同步梯度，但是teacher require_grad_(False)、eval() 且no_grad（），它没有梯度，也就是没有需要 all-reduce / synchronize 的梯度张量
- **student 用 DDP**：因为每张卡都在训练 student，需要同步梯度，保证参数更新一致。
- **teacher 不用 DDP**：因为每张卡只是各自拿一份相同的 teacher 做本地前向，算出本地 batch 的 `teacher_logits` 就够了，不存在“teacher 参数更新后要同步”的问题

4.teacher 完全不需要多卡吗？
- **不需要 DDP 包装**
- 但**仍然会在每个训练进程各自拥有一份 teacher 副本，并放到当前 device 上**
`teacher_model, _ = init_model(..., device=args.device)`

每个进程本地各放一份冻结的 teacher，用来给本进程这张卡上的 batch 计算 soft target；但这些 teacher 副本之间不做 DDP 梯度同步。
