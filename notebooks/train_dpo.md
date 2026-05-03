它在做MiniMind的DPO偏好对齐训练

在 **已经做过 SFT 的模型权重** 上，继续用 **偏好数据（chosen / rejected）** 做 **DPO（Direct Preference Optimization）训练**，让策略模型更倾向于生成“人类偏好”的回答，而不是只做普通的 next-token 监督学习

**训练过程**：
加载 SFT 后的初始模型权重 → 再复制一份作为冻结参考模型 → 读取 DPO 偏好数据（chosen / rejected）→ 分别计算 policy 与 reference 对 chosen/rejected 的打分差 → 按 DPO 公式计算 loss → 反向传播更新 policy 模型。

和train_full_sft.py的区别：
- `train_full_sft.py` 是“给定标准答案，让模型拟合答案 token
- `train_dpo.py` 是“给定一对回答：chosen 比 rejected 更好，让模型学会偏向 chosen


### 模块功能

**1.环境引导与导入**
```
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
...
warnings.filterwarnings('ignore')
```

**2.logits_to_log_probs**
```
def logits_to_log_probs(logits, labels):

    # logits shape: (batch_size, seq_len, vocab_size)

    # labels shape: (batch_size, seq_len)

    # log_probs shape: (batch_size, seq_len)

    log_probs = F.log_softmax(logits, dim=2)

    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)

    return log_probs_per_token
```

把模型输出的 `logits` 转成“标签 token 的对数概率”，也就是从每个位置整个词表分布中，取出真实标签对应的 log-prob

==Notice==：为什么用 `log_softmax` 而不是 `softmax`？
因为后面要累加概率、做差、算损失，用 log 概率数值更稳定，也符合语言模型里“序列概率 = token 概率连乘，log 后变成连加”的习惯。

`log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
这一句是从每个位置整个词表分布里，取出“真实 token”的对数概率。
- `labels` 原来是 `(B, T)`
- `labels.unsqueeze(2)` 变成 `(B, T, 1)`
- `torch.gather(..., dim=2, index=...)` 表示：在词表维度上，根据 label token id 抽取对应位置的 log-prob
- 最后 `squeeze(-1)` 去掉最后那个长度为 1 的维度，得到 `(B, T)`

**3.dpo_loss**
```
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):

    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)

    ref_log_probs = (ref_log_probs * mask).sum(dim=1)

    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

  

    # 将 chosen 和 rejected 数据分开

    batch_size = ref_log_probs.shape[0]

    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]

    reject_ref_log_probs = ref_log_probs[batch_size // 2:]

    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]

    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

  

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs

    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits)

    return loss.mean()
```
实现 DPO 的核心损失。它先把 token 级 log-prob 按 mask 聚合成“整条回答的总 log-prob”，再把 batch 前半部分视作 chosen，后半部分视作 rejected，最后构造策略模型相对参考模型的 preference margin，并用 `-logsigmoid(beta * logits)` 做优化


==Notice==：torch.nn.LogSigmoid
公式：
![[Pasted image 20260418172555.png]]
图像：
![[Pasted image 20260418172649.png]]



**4.train_epoch**

**5.`__main__` 中的初始化部分**

**6.训练循环与清理**
