# Agentic RL 工具调用训练过程记录

日期：2026-05-09

本文整理从 `tool_sft_v6_final` 到 Agentic RL / Dynamic RLVR / MiniMind 原始 Agent RL 数据转换的完整实验过程，重点记录每一阶段的目标、数据、训练命令、评测结果、失败分析和下一步决策。

## 1. 总体目标

本阶段目标不是直接追求单次 eval 涨分，而是建立一个可复现的工具调用闭环：

1. 模型根据 prompt 生成工具调用。
2. 本地工具环境执行工具并返回 observation。
3. 轨迹 trajectory 被 reward 函数评分。
4. SFT 或 GRPO 根据可验证 reward 更新模型。
5. 训练前后对 basic / hard / dynamic / converted math eval 做对比，确保不退化。

当前项目只使用本地可执行工具：

- `calculate_math`
- `text_length`
- `unit_converter`
- `get_current_time`
- `random_number`，仅在部分旧数据/干扰任务中使用

核心原则：

- 不把 eval 文件混入训练。
- 优先训练可执行、可验证、可复现的数据。
- basic / hard eval 用作回归测试。
- RL 只有在模型已经具备基本动作能力且 reward 有方差时才值得进入。

## 2. v6 SFT 修补阶段

### 2.1 初始失败分析

对 `tool_sft_v6_hard/tool_sft_v6_basic_guarded_failures.jsonl` 做 failure counts：

```json
{
  "num_failed_rows": 5,
  "failures": {
    "wrong_tool": 4,
    "observation_ignored": 3,
    "wrong_args": 1,
    "wrong_final_answer": 1,
    "overuse_tool": 1,
    "tool_call_without_final": 1
  }
}
```

主要失败样本：

- `math_004`：`2**10 + 24` 参数/答案错。
- `text_002`：文本长度任务误调用 `calculate_math`。
- `time_001/002/003`：时间任务误调用 `calculate_math`，且忽略 observation。

### 2.2 第一轮 repair SFT

构造 `dataset/tool_sft_v6_basic_repair.jsonl`，覆盖：

- `calculate_math`
- `text_length`
- `get_current_time`

每条失败类型重复多次，增强模型对正确工具和参数的记忆。

随后和已有 hard train / patch 数据混合训练得到：

```text
tool_sft_v6_hard_repair
```

hard guarded 结果：

```json
{
  "valid_rate": 1.0,
  "tool_acc": 1.0,
  "args_acc": 1.0,
  "obs_use_acc": 0.9933,
  "answer_acc": 0.98,
  "avg_reward": 2.959
}
```

### 2.3 时间 observation 修补

basic 仍剩 3 个时间类失败，failure 全是：

```json
{
  "observation_ignored": 3
}
```

原因：模型会调用 `get_current_time`，但 final answer 只说“如工具返回所示”，没有使用具体 datetime。

构造：

```text
dataset/tool_sft_v6_time_obs_repair.jsonl
```

每条时间样本把工具返回的 `datetime` 和 `timezone` 直接写入 final answer。

混合训练：

```bash
cat dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    > dataset/tool_sft_v6_final_repair_mixed.jsonl

cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v6_hard_repair \
  --save_weight tool_sft_v6_final \
  --data_path ../dataset/tool_sft_v6_final_repair_mixed.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --num_workers 0 \
  --save_interval 100
```

### 2.4 `tool_sft_v6_final` 结果

basic guarded：

```json
{
  "valid_rate": 1.0,
  "tool_acc": 1.0,
  "args_acc": 1.0,
  "obs_use_acc": 1.0,
  "answer_acc": 1.0,
  "avg_reward": 3.0
}
```

hard guarded：

```json
{
  "valid_rate": 1.0,
  "tool_acc": 1.0,
  "args_acc": 1.0,
  "obs_use_acc": 0.9933,
  "answer_acc": 0.98,
  "avg_reward": 2.9598
}
```

结论：

- `tool_sft_v6_final` 是当前最佳 SFT 基座。
- basic 满分，hard 接近满分。
- 后续 RL 不应再用 eval 文件训练。

## 3. Agentic RL v2：闭环验证

### 3.1 代码增强

在 `trainer/train_agent.py` 中增加：

- `--rollout_temperature`
- `--max_tool_turns`
- `--save_rollout_trace_path`
- reward group diagnostics：
  - `RewardMin`
  - `RewardMax`
  - `GrpStd`
  - `ZeroGrpRate`

同时让 rollout trajectory 带上：

- `task_id`
- `prompt`
- `category`
- expected 字段

新增测试覆盖：

- CLI 参数进入 rollout。
- task metadata 进入 trajectory。
- trace JSONL 写入。
- `AgentRLDataset` 读取 hard RL 格式。
- group reward stats 正确计算。

测试结果：

```text
36 passed
```

### 3.2 v2 hard RL 训练数据

训练数据：

```text
dataset/agent_rl_tooluse_hard_train.jsonl
```

包含 800 条 hard RL train：

- hard_multistep_text_math：160
- hard_multistep_math_unit：160
- hard_field_selection：160
- hard_distractor：160
- hard_no_tool：160

不使用：

- `evals/tool_eval.jsonl`
- `evals/tool_eval_hard.jsonl`

### 3.3 v2 RL 训练结果

训练时大部分 step 日志类似：

```text
Reward:3.0000
RewardMin:3.0000
RewardMax:3.0000
GrpStd:0.0000
ZeroGrpRate:1.000
AdvStd:0.0000
Loss:0.0000
```

trace 统计：

```text
rows: 3200
reward_dist: [(3.0, 3189), (0.3, 7), (1.4, 4)]
zero_group_steps: 794 / 800
rate: 0.9925
```

评测：

```json
{
  "model": "agent_tool_v2_hard",
  "hard_avg_reward": 2.9598,
  "hard_answer_acc": 0.98,
  "basic_failures": 0,
  "hard_failures": {
    "wrong_final_answer": 3,
    "observation_ignored": 1
  }
}
```

结论：

- v2 RL 闭环跑通且不退化。
- 但 SFT 基座太强，hard train 已饱和。
- `ZeroGrpRate=0.9925`，GRPO 基本没有有效 advantage。
- 不应继续在 `agent_rl_tooluse_hard_train.jsonl` 上训练。

## 4. Dynamic RLVR v3：制造 reward variance

### 4.1 设计目标

新增 dynamic 数据，目标是让 `tool_sft_v6_final` 初始准确率落在 55% 到 90%，从而让 RL 有 reward 方差。

新增数据生成器：

```text
scripts/make_agentic_tooluse_dynamic_data.py
```

新增输出：

- `evals/tool_eval_dynamic_l1.jsonl`
- `evals/tool_eval_dynamic_l2.jsonl`
- `evals/tool_eval_dynamic_l3.jsonl`
- `evals/tool_eval_dynamic_l4.jsonl`
- `dataset/agent_rl_tooluse_dynamic_train.jsonl`
- `dataset/tool_sft_dynamic_seed.jsonl`

覆盖四类：

- 3-step text -> math -> unit
- 3-step math -> unit -> math
- strong distractor
- adversarial no-tool-with-tools

reward 新增 `dense_v2`：

- 正确轨迹高分但不简单饱和。
- 惩罚 extra tool。
- 奖励 concise final。
- 支持 repeated tool expected args。

### 4.2 英文 dynamic 结果

`tool_sft_v6_final` 在英文 dynamic L1-L4 上很低：

```text
L1 answer_acc: 0.05
L2 answer_acc: 0.0375
L3 answer_acc: 0.0375
L4 answer_acc: 0.0375
```

bridge SFT 后 `tool_sft_v7_dynamic_seed`：

```text
basic reward: 3.0
hard reward: 2.9492
L1-L4 answer_acc: 0.25 左右
```

结论：

- 英文 dynamic 数据对中文为主的基座过难。
- 模型主要只学会了 no-tool 类。
- 需要把语言难度和任务结构难度拆开。

## 5. 中文/混合 dynamic 数据

### 5.1 代码改造

`make_agentic_tooluse_dynamic_data.py` 新增：

```bash
--language zh|en|mixed
```

中文输出：

- `evals/tool_eval_dynamic_zh_l1.jsonl`
- `evals/tool_eval_dynamic_zh_l2.jsonl`
- `evals/tool_eval_dynamic_zh_l3.jsonl`
- `evals/tool_eval_dynamic_zh_l4.jsonl`
- `dataset/tool_sft_dynamic_zh_seed.jsonl`
- `dataset/agent_rl_tooluse_dynamic_zh_train.jsonl`

每条 row 增加：

- `language`
- `difficulty_level`
- `reward_profile`
- expected 字段

### 5.2 中文 bridge SFT

训练：

```bash
cat dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_dynamic_zh_seed.jsonl \
    > dataset/tool_sft_v7_dynamic_zh_seed_mixed.jsonl

cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v6_final \
  --save_weight tool_sft_v7_dynamic_zh_seed \
  --data_path ../dataset/tool_sft_v7_dynamic_zh_seed_mixed.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 8e-7 \
  --num_workers 0 \
  --save_interval 100
```

结果：

```text
basic reward: 3.0
hard reward: 2.9303
dynamic zh L1 answer_acc: 0.2625
dynamic zh L2 answer_acc: 0.25
dynamic zh L3 answer_acc: 0.2625
dynamic zh L4 answer_acc: 0.25
```

结论：

- 中文数据比英文略好，但仍主要学到 no-tool。
- 3-step 工具链结构仍然没学会。
- 小规模 synthetic dynamic seed 不足以建立新工具链行为。

## 6. 转换 MiniMind 原始 Agent RL 数学数据

### 6.1 动机

原始 `dataset/agent_rl_math.jsonl` 有约 20,000 条，全部带 `gt`，且数学表达式大多可解析。

相比 synthetic dynamic，它有更多表达方式和更大规模，适合转成当前 `calculate_math` 工具任务。

不直接使用 `agent_rl.jsonl` 全量数据，因为包含天气、翻译、汇率等当前不可执行工具。

### 6.2 转换脚本

新增：

```text
scripts/convert_minimind_agent_rl_data.py
```

输出：

- `evals/tool_eval_minimind_math.jsonl`
- `dataset/tool_sft_minimind_math_seed.jsonl`
- `dataset/agent_rl_tooluse_minimind_math_train.jsonl`

转换规则：

- 只保留 final assistant 为空的样本。
- `gt` 必须非空。
- user prompt 中必须解析出数学表达式。
- 表达式数量必须等于 `gt` 数量。
- 用当前 `calculate_math` 执行，结果必须和 `gt` 匹配。
- 多表达式转成 repeated `calculate_math`：

```json
{
  "expected_tool_sequence": ["calculate_math", "calculate_math"],
  "expected_args": {
    "calculate_math": [
      {"expression": "1+1"},
      {"expression": "2*3"}
    ]
  }
}
```

测试：

```text
55 passed
```

### 6.3 转换统计

只使用 `agent_rl_math.jsonl`：

```text
source_rows: 20000
converted_rows: 19402
converted_math_single: 4508
converted_math_multi: 14894
eval rows: 1000
sft seed rows: 12000
rl train rows: 6402
prompt overlap: none
```

## 7. v8：原始数学数据 bridge SFT

### 7.1 训练

```bash
cat dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_dynamic_zh_seed.jsonl \
    dataset/tool_sft_minimind_math_seed.jsonl \
    > dataset/tool_sft_v8_math_dynamic_bridge_mixed.jsonl

cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v6_final \
  --save_weight tool_sft_v8_math_dynamic_bridge \
  --data_path ../dataset/tool_sft_v8_math_dynamic_bridge_mixed.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 3e-7 \
  --num_workers 0 \
  --save_interval 200
```

### 7.2 结果

```text
basic reward: 3.0
hard reward: 2.9033
minimind_math overall answer_acc: 0.227
dynamic zh L1 answer_acc: 0.3625
dynamic zh L2 answer_acc: 0.4125
dynamic zh L3 answer_acc: 0.30
dynamic zh L4 answer_acc: 0.3125
```

按 single / multi 拆分：

```text
converted_math_single:
  tool_acc: 0.8992
  args_acc: 1.0
  answer_acc: 0.9538
  reward: 2.4689

converted_math_multi:
  tool_acc: 0.0
  args_acc: 0.0
  answer_acc: 0.0
  reward: -1.7483
```

结论：

- 单表达式数学已经学会。
- 多表达式 repeated `calculate_math` 完全没学会。
- 后续应专门补 repeated same-tool multi-call，而不是继续扩大普通数学数据。

## 8. v9：normalized multi 数据

### 8.1 代码改造

`convert_minimind_agent_rl_data.py` 新增：

```bash
--prompt_mode original|normalized|both
--include_single
--include_multi
--max_expressions
```

normalized multi prompt 示例：

```text
Call the calculator tool once for each expression, in order, then give the results in the same order: 1+1; 2*3; 9-4.
```

生成 multi-only seed：

```bash
python scripts/convert_minimind_agent_rl_data.py \
  --skip_agent_rl \
  --prompt_mode normalized \
  --include_multi \
  --sft_seed_size 6000 \
  --sft_seed_path dataset/tool_sft_minimind_math_normalized_seed.jsonl \
  --eval_path evals/tool_eval_minimind_math_normalized.jsonl \
  --rl_train_path dataset/agent_rl_tooluse_minimind_math_normalized_train.jsonl
```

统计：

```text
verified rows: 14884 {'converted_math_multi': 14884}
eval rows: 1000
sft seed rows: 6000
rl train rows: 7884
```

### 8.2 v9 训练

```bash
cat dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_dynamic_zh_seed.jsonl \
    dataset/tool_sft_minimind_math_normalized_seed.jsonl \
    > dataset/tool_sft_v9_repeated_math_bridge_mixed.jsonl

cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v6_final \
  --save_weight tool_sft_v9_repeated_math_bridge \
  --data_path ../dataset/tool_sft_v9_repeated_math_bridge_mixed.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 3e-7 \
  --num_workers 0 \
  --save_interval 200
```

### 8.3 v9 结果

```text
basic reward: 3.0
hard reward: 2.95
minimind_math_normalized:
  valid: 1.0
  tool_acc: 0.0
  args_acc: 0.0
  answer_acc: 0.001
  reward: -1.799
```

结论：

- basic / hard 没伤。
- normalized multi 仍完全没学会。
- 怀疑训练序列被截断：默认 `max_seq_len=768`，多轮 tool-call 样本包含 tools schema + 多个 assistant/tool turn，后半段可能被截掉。

## 9. v10：长上下文 repeated math

### 9.1 训练

关键变化：

```bash
--max_seq_len 1536
--batch_size 2
```

训练：

```bash
cat dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_dynamic_zh_seed.jsonl \
    dataset/tool_sft_minimind_math_normalized_seed.jsonl \
    > dataset/tool_sft_v10_repeated_math_longctx_mixed.jsonl

cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v6_final \
  --save_weight tool_sft_v10_repeated_math_longctx \
  --data_path ../dataset/tool_sft_v10_repeated_math_longctx_mixed.jsonl \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 2e-7 \
  --max_seq_len 1536 \
  --num_workers 0 \
  --save_interval 300
```

### 9.2 v10 结果

```text
basic:
  valid: 0.9697
  tool: 0.9697
  args: 1.0
  answer: 1.0
  reward: 2.9394

hard:
  valid: 0.8867
  tool: 0.8867
  args: 0.9933
  answer: 0.9933
  reward: 2.8233

minimind_math_normalized:
  valid: 0.689
  tool: 0.689
  args: 0.988
  answer: 0.966
  reward: 1.9955
```

结论：

- 长上下文确实解决 repeated tool-call 学习问题。
- normalized multi 从几乎 0 提升到 `answer_acc=0.966`。
- 但旧 basic/hard 工具行为明显退化，尤其 hard `valid/tool` 掉到 0.8867。
- 原因是 6000 条 normalized multi seed 配比太重，且长上下文训练改变了旧策略。

## 10. 当前下一步：v11 平衡修复

当前不应进入 RL。

原因：

- v10 虽然学会了 multi repeated math，但 old hard 已明显退化。
- RL 应基于一个稳定、不退化的 SFT checkpoint。
- 如果在 v10 上直接 RL，可能进一步放大新分布偏差。

### 10.1 v11 目标

从 v10 继续做低学习率 SFT 修复：

- 恢复 basic/hard。
- 保留 repeated math multi 能力。
- 降低 normalized multi 数据比例。

目标指标：

```text
basic avg_reward 接近 3.0
hard avg_reward >= 2.94
minimind_math_normalized tool_acc >= 0.60
minimind_math_normalized answer_acc >= 0.90
```

### 10.2 v11 推荐命令

生成 2000 条 normalized multi seed：

```bash
cd /root/minimind

python scripts/convert_minimind_agent_rl_data.py \
  --skip_agent_rl \
  --prompt_mode normalized \
  --include_multi \
  --sft_seed_size 2000 \
  --sft_seed_path dataset/tool_sft_minimind_math_normalized_seed_2k.jsonl \
  --eval_path evals/tool_eval_minimind_math_normalized_2k_split.jsonl \
  --rl_train_path dataset/agent_rl_tooluse_minimind_math_normalized_2k_train.jsonl
```

拼接训练集：

```bash
cat dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_small_v4_patch.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_hard_train.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_v6_time_obs_repair.jsonl \
    dataset/tool_sft_dynamic_zh_seed.jsonl \
    dataset/tool_sft_minimind_math_normalized_seed_2k.jsonl \
    > dataset/tool_sft_v11_balanced_repeated_math_mixed.jsonl
```

从 v10 继续低学习率训练：

```bash
cd /root/minimind/trainer

OMP_NUM_THREADS=1 python train_full_sft.py \
  --from_weight tool_sft_v10_repeated_math_longctx \
  --save_weight tool_sft_v11_balanced_repeated_math \
  --data_path ../dataset/tool_sft_v11_balanced_repeated_math_mixed.jsonl \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 1e-7 \
  --max_seq_len 1536 \
  --num_workers 0 \
  --save_interval 300
```

评测：

```bash
cd /root/minimind
mkdir -p evals/reports/tool_sft_v11_balanced_repeated_math

OMP_NUM_THREADS=1 python evals/run_tool_eval.py \
  --weight tool_sft_v11_balanced_repeated_math \
  --mode guarded \
  --eval_path evals/tool_eval.jsonl \
  --device cuda \
  --continue_on_error \
  --output evals/reports/tool_sft_v11_balanced_repeated_math/basic_guarded.json

OMP_NUM_THREADS=1 python evals/run_tool_eval.py \
  --weight tool_sft_v11_balanced_repeated_math \
  --mode guarded \
  --eval_path evals/tool_eval_hard.jsonl \
  --device cuda \
  --continue_on_error \
  --output evals/reports/tool_sft_v11_balanced_repeated_math/hard_guarded.json

OMP_NUM_THREADS=1 python evals/run_tool_eval.py \
  --weight tool_sft_v11_balanced_repeated_math \
  --mode guarded \
  --eval_path evals/tool_eval_minimind_math_normalized.jsonl \
  --device cuda \
  --continue_on_error \
  --output evals/reports/tool_sft_v11_balanced_repeated_math/minimind_math_normalized_guarded.json
```

汇总：

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("evals/reports/tool_sft_v11_balanced_repeated_math")
for name in ["basic", "hard", "minimind_math_normalized"]:
    r = json.loads((base / f"{name}_guarded.json").read_text(encoding="utf-8"))
    print(
        name,
        "valid", round(r["valid_rate"], 4),
        "tool", round(r["tool_acc"], 4),
        "args", round(r["args_acc"], 4),
        "ans", round(r["answer_acc"], 4),
        "reward", round(r["avg_reward"], 4),
    )
PY
```

## 11. 关键经验

### 11.1 RL 不是万能补丁

v2 hard RL 显示，如果 SFT 已经接近满分，reward 没有方差：

```text
ZeroGrpRate 接近 1
AdvStd 接近 0
Loss 接近 0
```

这种情况下继续 RL 基本学不到东西。

### 11.2 数据难度必须落在中间区间

英文 dynamic L1-L4 对中文基座过难，answer_acc 只有 0.04 左右。

过难数据不能直接 RL，因为 rollout 几乎全错，reward 信号不稳定。

### 11.3 repeated same-tool multi-call 是新能力

单次 `calculate_math` 很容易学会，但连续多次：

```text
assistant tool_call
tool observation
assistant tool_call
tool observation
assistant final
```

这是一个新的行为模式，不能只靠普通数学 SFT 泛化出来。

### 11.4 长上下文是 multi-turn SFT 的必要条件

v9 默认 `max_seq_len=768` 时 normalized multi 几乎为 0。

v10 `max_seq_len=1536` 后 normalized multi：

```text
answer_acc: 0.966
```

这说明多轮 tool-call 样本在短上下文下很可能被截断。

### 11.5 新能力和旧能力需要配比平衡

v10 学会 repeated math，但 old hard 退化。

后续应通过：

- 降低新数据比例。
- 重复旧 hard/basic tool 数据。
- 从 v10 做低学习率修复。
- 每轮都评 basic/hard。

## 12. 当前 checkpoint 定位

| checkpoint | 定位 | 是否推荐继续 |
|---|---|---|
| `tool_sft_v6_final` | 最佳稳定 SFT 基座，basic 满分，hard 接近满分 | 推荐作为安全起点 |
| `agent_tool_v2_hard` | RL 闭环验证成功，但 reward 饱和 | 保留阶段成果，不继续训练 |
| `tool_sft_v7_dynamic_zh_seed` | 中文 dynamic bridge，提升有限，hard 略退化 | 不推荐继续 |
| `tool_sft_v8_math_dynamic_bridge` | 原始 math 转换尝试，发现 single/multi 差异 | 诊断用 |
| `tool_sft_v9_repeated_math_bridge` | normalized multi + 短上下文，multi 未学会 | 诊断用 |
| `tool_sft_v10_repeated_math_longctx` | multi 学会，但 old hard 退化 | 可作为 v11 修复起点 |
| `tool_sft_v11_balanced_repeated_math` | 下一步目标：平衡旧能力和 repeated multi | 待训练/评测 |

## 13. 进入 RL 的条件

只有当 v11 或后续 checkpoint 满足以下条件，才进入 RL：

```text
basic avg_reward >= 2.95
hard avg_reward >= 2.94
hard answer_acc >= 0.97
minimind_math_normalized answer_acc >= 0.90
malformed_rate = 0
loop_rate = 0
unfinished_rate = 0
```

并且 RL rollout trace 应满足：

```text
ZeroGrpRate < 0.70
reward 分布不集中在单一分值
failure_counts 中 wrong_args / wrong_final_answer 有下降空间
```

若 `ZeroGrpRate` 仍很高，说明 SFT 已饱和，应继续调整数据难度，而不是调学习率。

## 14. 推荐下一步

立即执行：

1. 训练 `tool_sft_v11_balanced_repeated_math`。
2. 评测 basic / hard / minimind_math_normalized。
3. 如果 old hard 恢复且 repeated math 保留，再考虑 RL smoke。
4. 如果 old hard 仍低于 2.94，继续降低 normalized multi seed 数量到 1000，或加大 hard/basic 重复权重。

暂不执行：

- 不直接从 v10 做 RL。
- 不把 `evals/tool_eval*.jsonl` 混入训练。
- 不扩展天气、翻译、汇率等外部工具。
- 不继续扩大英文 dynamic 数据。
