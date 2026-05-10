# MiniMind Agentic RL 算法选择说明

本文说明为什么当前 Tool-Use Agentic RL 路线选择 `SFT -> GRPO/RLVR`，而不是只做 SFT、DPO 或完整 PPO。

## 1. 为什么不能只做 SFT

Tool-call SFT 适合让模型学会标准格式：

```text
user -> assistant tool_call -> tool_response -> assistant final
```

它主要提升：

```text
valid_rate
tool_acc
args_acc
```

但 SFT 不擅长优化这些动态行为：

```text
1. 工具调用后是否真正使用 observation
2. 是否避免重复调用同一个工具
3. 是否在合适时机停止 tool_call 并给 final answer
4. 多步工具链中到底是哪一步失败
5. 工具结果正确但模型最终回答改错的情况
```

因此本项目把 SFT 定位为“格式和基础工具选择补强”，把 RLVR/GRPO 定位为“基于环境反馈优化行为”。

## 2. 为什么 DPO 不是主路线

DPO 适合静态偏好数据：

```text
prompt + chosen_response + rejected_response
```

但当前项目是动态工具环境：

```text
prompt
  -> tool_call
  -> tool_response
  -> final answer
```

DPO 的局限是：

```text
1. 不天然执行工具
2. 更依赖离线偏好对，不适合在线 rollout
3. 难以处理工具 observation 改变后续路径的情况
4. 不方便单独评价中间 action 的 tool_name、args、loop、observation_use
```

DPO 可以作为后续 baseline，但不是当前 Agentic RL 主路线。

## 3. 为什么不优先做完整 PPO

PPO 更标准，但工程成本更高：

```text
1. 需要 actor + critic
2. 需要训练 value function
3. 需要 GAE 或类似优势估计
4. 小模型小数据下 critic 估计容易不稳定
5. 显存和实现复杂度更高
```

MiniMind 当前是轻量模型和小规模可验证工具任务。优先做 GRPO/RLVR 更适合用较低工程成本打通闭环。

## 4. 为什么选择 GRPO / RLVR

当前任务天然适合 GRPO/RLVR：

```text
1. 工具执行结果可以提供规则 reward
2. 同一个 prompt 可以采样多条 trajectory
3. 可以用组内 reward 均值和方差估计相对 advantage
4. 不需要额外 critic
5. 更适合小模型、低成本实验和快速 ablation
```

核心形式：

```text
对同一个 task 采样 G 条 trajectory:
y_1, y_2, ..., y_G

ToolUseEnv 执行后得到 reward:
r_1, r_2, ..., r_G

组内 advantage:
A_i = (r_i - mean(r)) / (std(r) + eps)
```

然后提高高 reward trajectory 的 token 概率，压低低 reward trajectory 的 token 概率，并通过 reference KL 控制策略漂移。

## 5. 当前实现中的稳定性监控

`trainer/train_agent.py` 现在记录：

```text
reward mean
group reward std
KL
clip_frac
avg_response_len
tool_call_count
valid/tool_acc/args_acc/obs_use/answer_acc
loop/malformed/unfinished
```

如果 `group_reward_std` 过低，会打印警告：

```text
Low reward variance: GRPO/CISPO advantage may collapse
```

这是因为 GRPO 依赖组内 reward 差异。如果同组采样 reward 几乎相同，advantage 信号会塌缩，训练很难学到有效行为。

