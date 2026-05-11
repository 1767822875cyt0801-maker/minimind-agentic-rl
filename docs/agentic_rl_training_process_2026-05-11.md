# Agentic RL 训练过程记录

日期：2026-05-11

本文记录 5.11 在 MiniMind Tool-Use Agentic RL 闭环上的继续推进：修复 AutoDL 新容器依赖问题，完成 `agent_tool_v13_rl_norm_stage3`，并确定下一步进入 2048-row stage4，而不是直接全量 RL。

## 1. AutoDL 依赖问题

运行 stage3 时，新 AutoDL 容器在导入 `transformers` 阶段报错：

```text
ValueError: All ufuncs must have type `numpy.ufunc`
```

堆栈路径是：

```text
transformers -> sklearn -> scipy.special
```

因此这不是 RL 脚本逻辑问题，而是 `numpy/scipy/sklearn` 二进制依赖组合不兼容。

处理方式：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python -m pip install --no-cache-dir --force-reinstall \
  -i https://mirrors.aliyun.com/pypi/simple \
  --trusted-host mirrors.aliyun.com \
  "numpy==1.26.4" \
  "scipy==1.12.0" \
  "scikit-learn==1.5.1"
```

同时在 `requirements.txt` 中补充：

```text
scipy==1.12.0
```

并在公共 RL stage runner `run_v13_rl_stage1_autodl.sh` 中加入 Python 依赖预检。因为 stage2/stage3/stage4 都复用这个 runner，以后新容器如果再遇到类似问题，会在训练前直接给出诊断。

## 2. Stage3 RL

### 2.1 动机

Stage1 和 stage2 都从干净的 v13 SFT checkpoint 出发：

```text
tool_sft_v13_stop_repair_repeated_math
```

并且都证明了：

- basic/hard 不退化；
- normalized multi 有小幅正收益；
- loop 没有恶化；
- reward 增益可以随数据规模扩大而继续增加。

因此 stage3 的目标是把 RL 数据从 512 rows 扩大到 1024 rows，验证这种趋势是否仍然成立。

### 2.2 配置

使用脚本：

```text
run_v13_rl_stage3_autodl.sh
```

默认配置：

```text
STAGE_LABEL=stage3
FROM_WEIGHT=tool_sft_v13_stop_repair_repeated_math
SAVE_WEIGHT=agent_tool_v13_rl_norm_stage3
RL_ROWS=1024
NUM_GENERATIONS=8
ROLLOUT_TEMPERATURE=1.0
MAX_TOOL_TURNS=4
RL_LEARNING_RATE=5e-8
```

注意：stage3 仍从干净 v13 SFT checkpoint 开始，不从 stage2 checkpoint 继续。这样 stage1、stage2、stage3 是同一起点、不同 RL 数据规模的可比实验。

## 3. Stage3 结果

Normalized eval：

```json
{
  "model": "agent_tool_v13_rl_norm_stage3",
  "mode": "guarded",
  "num_cases": 1000,
  "valid_rate": 0.941,
  "tool_acc": 0.941,
  "args_acc": 0.99,
  "obs_use_acc": 1.0,
  "answer_acc": 0.962,
  "loop_rate": 0.024,
  "malformed_rate": 0.0,
  "unfinished_rate": 0.0,
  "avg_tool_calls": 2.717,
  "avg_reward": 2.5776
}
```

与 v13 baseline 对比：

```text
basic:
  reward: 3.0000 -> 3.0000
  loop: 0.0000 -> 0.0000

hard:
  reward: 2.9820 -> 2.9820
  answer: 0.9933 -> 0.9933
  loop: 0.0000 -> 0.0000

minimind_math_normalized:
  valid/tool: 0.9310 -> 0.9410  (+0.0100)
  args: 0.9890 -> 0.9900  (+0.0010)
  answer: 0.9590 -> 0.9620  (+0.0030)
  loop: 0.0260 -> 0.0240  (-0.0020)
  reward: 2.5491 -> 2.5776  (+0.0285)
```

Gate：

```text
STAGE3_PASS: 1
```

## 4. 趋势判断

目前三个 RL stage 的 normalized 指标趋势是：

| checkpoint | RL rows | valid/tool | answer | loop | avg_reward | 相对 v13 reward |
|---|---:|---:|---:|---:|---:|---:|
| `tool_sft_v13_stop_repair_repeated_math` | 0 | 0.931 | 0.959 | 0.026 | 2.5491 | baseline |
| `agent_tool_v13_rl_norm_stage1` | 128 | 0.933 | 0.959 | 0.025 | 2.5537 | +0.0046 |
| `agent_tool_v13_rl_norm_stage2` | 512 | 0.936 | 0.962 | 0.024 | 2.5676 | +0.0185 |
| `agent_tool_v13_rl_norm_stage3` | 1024 | 0.941 | 0.962 | 0.024 | 2.5776 | +0.0285 |

结论：

- RL 的收益不是一次性噪声，stage1 到 stage3 连续同方向；
- basic/hard 完全没有被 normalized multi RL 破坏；
- valid/tool 仍在提升；
- answer 和 loop 从 stage2 到 stage3 开始接近平台期；
- 当前更像“稳定小收益放大”，还不是“大幅能力跃迁”。

这说明 Agentic RL 闭环已经成立，但仍应该继续按 stage 放大，而不是直接进入长训。

## 5. 下一步

先检查 stage3 rollout trace：

```bash
cd /root/minimind

python - <<'PY'
import json
from pathlib import Path

p = Path("evals/reports/agent_tool_v13_rl_norm_stage3/rl_stage3_trace_summary.json")
r = json.loads(p.read_text(encoding="utf-8"))
for k in ["reward_min", "reward_max", "reward_mean", "zero_group_rate"]:
    print(k, r.get(k))
print("failure_counts", r.get("failure_counts", [])[:10])
PY
```

如果 `zero_group_rate` 仍明显低于 `0.70`，并且 failures 没有出现明显异常，就运行 stage4：

```bash
cd /root/minimind
bash run_v13_rl_stage4_autodl.sh
```

Stage4 默认配置：

```text
FROM_WEIGHT=tool_sft_v13_stop_repair_repeated_math
SAVE_WEIGHT=agent_tool_v13_rl_norm_stage4
RL_ROWS=2048
NUM_GENERATIONS=8
ROLLOUT_TEMPERATURE=1.0
MAX_TOOL_TURNS=4
RL_LEARNING_RATE=5e-8
```

Stage4 仍从干净 v13 SFT checkpoint 开始，不从 stage3 checkpoint 继续。

## 6. 暂不执行

暂时不要做：

- 不直接全量 RL；
- 不从 `agent_tool_v13_rl_norm_stage3` 续训，除非单独设计 continuation 实验；
- 不混入 eval 文件；
- 不改工具集合；
- 不把 `MAX_TOOL_TURNS` 改回 3；
- 不提高学习率来追求更大短期提升。

当前最稳路线是：

```text
v13 SFT baseline
  -> stage1 128 rows
  -> stage2 512 rows
  -> stage3 1024 rows
  -> stage4 2048 rows
  -> 再决定是否做 filtered hard-normalized RL 或 continuation RL
```
