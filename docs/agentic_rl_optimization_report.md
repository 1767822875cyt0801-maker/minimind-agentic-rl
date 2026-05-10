# Agentic RL 面试建议判断与本轮优化说明

## 1. 对建议的判断

这份面试建议总体是正确的。

它没有否定当前项目，而是指出当前项目虽然已经有：

```text
agent/tools.py
agent/parser.py
agent/tool_env.py
agent/trajectory.py
agent/reward.py
evals/run_tool_eval.py
dataset/tool_sft_small.jsonl
dataset/agent_rl_tooluse.jsonl
trainer/train_agent.py 接入 ToolUseEnv
```

但还需要补足“面试硬证据”：

```text
1. 为什么选 GRPO，而不是 PPO / DPO
2. reward 是否会被 hacking
3. sparse reward 和 reward 方差如何监控
4. tool_call 后模型不 final 怎么诊断
5. raw / guarded 差距说明什么
6. 有没有 ablation 和 failure taxonomy
7. loss mask 是否避免训练 tool_response
```

这些问题都是 Agentic RL 和 Agent 工程化中真实会被追问的点，所以建议方向正确。

## 2. 本轮继续优化了什么

本轮没有推翻已有架构，而是在 P0 方向补强：

```text
1. 增加训练稳定性监控
2. 增加 reward breakdown 聚合
3. 增加 response mask 辅助检查
4. 增加 failure-mode 评测集
5. 增加 ablation report 脚本
6. 增加算法选择说明文档
```

## 3. 代码改动明细

### 3.1 `agent/reward.py`

新增：

```python
aggregate_reward_infos(reward_infos)
```

作用：

```text
把每条 trajectory 的 reward_info 聚合成 batch 级别指标，
供 train_agent.py 记录训练稳定性和行为指标。
```

同时新增 failure type：

```text
tool_call_without_final
```

触发条件：

```text
已经有工具 observation，但没有 assistant_final。
```

它用于诊断“模型一直 tool_call，不肯最终回答”的问题。

### 3.2 `trainer/train_agent.py`

增强 `calculate_rewards()`：

```text
支持 return_infos=True
返回 rewards + reward_infos
```

训练日志新增：

```text
ClipFrac
ToolCalls
Valid
ToolAcc
ArgsAcc
ObsUse
AnsAcc
Loop
Malformed
```

同时新增：

```text
--reward_std_warn_threshold
```

当组内 reward 方差过低时打印警告，提示 GRPO/CISPO advantage 可能塌缩。

### 3.3 `agent/masks.py`

新增渲染后 chat span 分类工具：

```python
classify_rendered_chat_spans(rendered_text)
build_rendered_char_action_mask(rendered_text)
```

作用：

```text
辅助检查哪些片段属于模型 action，哪些只是上下文或工具 observation。
```

期望：

```text
system/tools schema -> 不训练
user prompt -> 不训练
assistant tool_call -> 训练
tool_response -> 不训练
assistant final -> 训练
```

### 3.4 `test_agent_response_mask.py`

新增测试：

```text
验证 system/user/tool_response 不标为 trainable，
assistant tool_call 和 assistant final 标为 trainable。
```

这对应面试建议中的 response mask 检查。

### 3.5 `evals/tool_eval_failure_modes.jsonl`

新增 failure-mode 评测集，覆盖：

```text
unnecessary_tool
wrong_tool_trap
wrong_args_trap
reward_hacking_trap
tool_error
tool_loop_risk
observation_ignored
tool_call_without_final
```

作用：

```text
专门检查 reward hacking、错误工具、错误参数、忽略 observation、
工具调用后不 final 等高风险行为。
```

### 3.6 `evals/run_ablation.py`

新增 ablation 汇总脚本，支持：

```bash
python evals/run_ablation.py --ablation raw_guarded_gap \
  --raw evals/reports/full_sft_raw.json \
  --guarded evals/reports/full_sft_guarded.json

python evals/run_ablation.py --ablation reward_components \
  --reports evals/reports/full_sft_raw_trajectories.jsonl

python evals/run_ablation.py --ablation failure_counts \
  --reports evals/reports/full_sft_raw_failures.jsonl
```

作用：

```text
把 raw/guarded 差距、reward component 均值、failure taxonomy 统计成报告。
```

### 3.7 `docs/rl_algorithm_choice.md`

新增算法选择说明，回答：

```text
为什么不是纯 SFT
为什么 DPO 不是主路线
为什么不优先完整 PPO
为什么当前选择 GRPO / RLVR
```

## 4. 文件之间的关系

### 4.1 训练链路

```text
dataset/agent_rl_tooluse.jsonl
  -> AgentRLDataset
  -> trainer/train_agent.py
  -> rollout_single()
  -> ToolUseEnv.step()
      -> parser.py 解析 tool_call
      -> tools.py 执行工具
      -> trajectory.py 记录轨迹
  -> reward.py compute_total_reward()
  -> aggregate_reward_infos()
  -> GRPO/CISPO loss
```

### 4.2 评测链路

```text
evals/tool_eval.jsonl 或 evals/tool_eval_failure_modes.jsonl
  -> evals/run_tool_eval.py
  -> ToolUseEnv
  -> compute_total_reward
  -> evals/reports/*.json
  -> evals/run_ablation.py
```

### 4.3 SFT 链路

```text
dataset/tool_sft_small.jsonl
  -> SFTDataset.create_chat_prompt()
  -> tokenizer.apply_chat_template(..., tools=tools)
  -> train_full_sft.py
```

## 5. 为什么这些优化有价值

### 5.1 算法选择更能讲清楚

以前只能说：

```text
我接了 GRPO/RLVR
```

现在可以说：

```text
DPO 不适合动态工具执行；
PPO 成本高且需要 critic；
GRPO 适合小模型和可验证工具环境。
```

### 5.2 训练稳定性更可诊断

新增指标能回答：

```text
reward 是否真的上升？
reward 方差是否塌缩？
KL 是否爆炸？
模型是否疯狂调用工具？
模型是否格式正确但答案不对？
```

### 5.3 failure taxonomy 更像真实项目

新增 failure-mode 数据后，可以专门展示：

```text
wrong_tool
wrong_args
observation_ignored
tool_loop
tool_call_without_final
unnecessary_tool
```

这比只看 answer_acc 更接近真实 Agent 迭代。

## 6. 后续仍可继续优化的地方

建议后续按优先级继续做：

```text
P1:
1. multi-step eval：random_number -> calculate_math
2. unseen tool 泛化：工具改名 / 新工具 schema
3. tool timeout / latency / tool_error 统一返回结构
4. state_store：保存和恢复 session trajectory

P2:
1. tool_router：工具数量变多时选 top-k schema
2. web demo 展示 tool_call / observation / reward breakdown
3. SQLite 实验库 evals/reports/agent_runs.sqlite
```

## 7. Agentic RL 完成线

完整完成线应该是：

```text
Stage 1：统一 Agent runtime
Stage 2：raw/guarded baseline eval
Stage 3：tool-call SFT
Stage 4：GRPO/RLVR 训练
Stage 5：训练稳定性监控
Stage 6：failure-mode eval 和 ablation
Stage 7：multi-step / unseen tool / tool scaling
Stage 8：项目报告和面试展示
```

当前已经完成 Stage 1-6 的主体骨架。下一步重点是 Stage 7：多步任务、未见工具泛化和工具数量扩展。

