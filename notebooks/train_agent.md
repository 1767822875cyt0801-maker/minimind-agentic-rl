“Agent 工具使用能力”的强化学习训练脚本
在一个已经具备基本对话能力的模型基础上，继续通过 **rollout → 工具调用 → 奖励计算 → 策略更新** 这条链，把模型往“会思考、会调用工具、能根据工具结果继续回答”的方向再训练一层
它用强化学习的方法训练 MiniMind，让模型不仅会回答问题，还会按需要调用工具、接收工具返回结果、继续多轮推理，并通过奖励函数把这种行为优化出来。


## 和其余几个文件的关系：
1.model_minimind.py

**模型长什么样**，由 `model_minimind.py` 定义；**怎么拿这个模型做 Agent 强化学习**，由 `train_agent.py` 决定

train_agent.py还会建立两份模型：
- `model`：当前正在被训练的策略模型
- `ref_model`：参考模型，不参与训练，只用于算 KL 约束

这和 RLHF / GRPO / PPO 里的“policy model + reference model”是同一种思想

2.lm_dataset.py
这个脚本从 `dataset.lm_dataset` 里导入了 `AgentRLDataset`。这个数据集中的一个batch里的每个样本至少包含三类字段：
- `messages`：对话消息列表
- `tools`：这个样本可用的工具定义
- `gt`：ground truth，奖励判断时要核对的目标答案或关键值

3.trainer_utils.py
trainer_utils.py是训练公共基建库

4.train_agent.py
- `train_agent.py` 负责“训练算法”
- `rollout_engine.py` 负责“把策略拿去跑一遍，收集轨迹”

## 模块功能
### 第一部分：环境准备与依赖导入

这一部分就是开头：

- `sys.path` / `__package__`
- 各种 Python 标准库
- PyTorch / distributed / optimizer / dataloader
- transformers tokenizer
- MiniMind 模型
- AgentRLDataset
- trainer_utils
- rollout_engine

### 作用

解决两个问题：

1. 让这个脚本能找到项目里的其他模块
2. 把训练所需的一切组件都导进来


---

### 第二部分：工具定义 + 模拟环境 + 工具执行器

这一块包括：

- `rep_penalty`
- `TOOLS`
- `WEATHER_DATA / TIME_DATA / EXCHANGE_DATA / TRANSLATE_DATA / UNIT_DATA`
- `MOCK_RESULTS`
- `CHECK_ARGS`
- `parse_tool_calls`
- `execute_tool`

### 作用

这一部分构建了一个 **简化版工具世界**：

- 模型能调用什么工具
- 这些工具长什么接口
- 调用后返回什么
- 参数是否合法
- 回答是否太重复

### 为什么它重要

因为 Agent RL 不是纯文本 RL，它需要一个 environment。  
这一块就是这个 environment 的最小实现。

### 本质

它把“工具调用”从抽象概念变成了可执行训练反馈。

---

### 第三部分：多轮 rollout 逻辑

这一块包括：

- `rollout_single`
- `rollout_batch`

#### `rollout_single` 在做什么

对一个样本跑一条轨迹：

1. 根据 `messages + tools` 拼 prompt
2. 用 rollout engine 生成一轮回答
3. 提取 tool call
4. 执行工具
5. 把工具结果追加回消息
6. 再继续下一轮
7. 最后返回：
    - final_output
    - final_context
    - prompt_ids
    - response_ids
    - response_mask
    - response_old_logps
    - 每轮输出
    - unfinished 标记

#### `rollout_batch` 在做什么

把 `rollout_single` 扩展到 batch，并且每个 prompt 还能采样 `num_gen` 个候选。

#### 这一部分的本质

这部分就是 **采样轨迹收集器**。  
在 RL 里，这一步相当于“让当前策略去环境里跑，收集 experience”。

### 第四部分：奖励计算

这一块核心是：

- `validate_gt_in_text`
- `calculate_rewards`

#### `validate_gt_in_text`

负责检查模型最终文本里是否命中了 GT：

- 可以直接字符串匹配
- 也可以把数字抽出来对比

#### `calculate_rewards`

负责给每条 completion 打分。

它的逻辑你可以记成两大分支：

##### A. 没调用工具

更偏向格式奖励 + RM 奖励 + 重复惩罚

##### B. 调了工具

更偏向：

- 工具选择对不对
- 参数合不合理
- 工具调用数量对不对
- 最终答案是否包含目标信息
- 是否未完成
- 是否重复

#### 这一部分的本质

它定义了“什么样的 agent 行为是好的”。

换句话说：  
**这部分决定了模型最终会学成什么样。**  
因为 RL 学到的不是数据本身，而是 reward 偏好。

---

### 第五部分：单个 epoch 的 RL 训练主循环

这一块是全文件最核心的函数：

- `rl_train_epoch(...)`

你可以把它理解成整个 Agent RL 的发动机。

#### 它做了什么

每个 batch 大致分 8 步：

##### 1）从 dataloader 取出训练样本

拿到：

- `messages_batch`
- `tools_batch`
- `gt_batch`

##### 2）先 rollout

用当前策略模型对每个 prompt 采样多个 completion。  
这是“收集经验”。

##### 3）把 prompt 和 response 打包成训练输入

包括：

- `input_ids`
- `prompt_lens`
- `full_response_masks`
- `old_per_token_logps`

这里的关键是区分：

- prompt token：不参与策略梯度
- 真正模型生成的 response token：参与 loss
- 工具观察产生的 token：mask 为 0，不直接参与策略更新

##### 4）用当前模型重新算 per-token logps

这是新策略下的 log probability。

##### 5）用参考模型算 ref logps

用于 KL 惩罚，防止策略漂移过大。

##### 6）算 reward，再算 advantage

它把同一 prompt 下的多个候选分成一组，做组内标准化：

advantages = (rewards - mean_r) / (std_r + 1e-4)

这正是 GRPO 那类方法的核心味道。

##### 7）算 policy loss

支持两种：

- `grpo`
- `cispo`

同时加上 KL 项和可选的 `aux_loss`（MoE 时）。

##### 8）反向传播、梯度累积、保存 checkpoint、记录日志

包括：

- `loss.backward()`
- grad clip
- optimizer step
- scheduler step
- rollout engine policy update
- save weight
- `lm_checkpoint(...)`

### 这一部分的本质

就是把：  
**rollout 结果 + reward + KL + old/new logps**  
拼成一个完整的 RL 优化闭环。

---

### 第六部分：`main` 主程序入口

也就是 `if __name__ == "__main__":` 下面整块。

这部分又可以再拆成几个子阶段：

#### 6.1 参数定义

`argparse` 定义了所有训练配置，比如：

- 训练轮数
- batch size
- 学习率
- generation 数量
- max_gen_len
- beta
- loss_type
- sglang 配置
- reward model 路径
- checkpoint 恢复等

#### 6.2 分布式与随机种子初始化

- `init_distributed_mode()`
- `setup_seed(...)`

#### 6.3 建立模型配置与 checkpoint 恢复

- 构造 `MiniMindConfig`
- 如果 `from_resume=1`，则读 checkpoint

#### 6.4 建立 autocast 环境

根据设备和 dtype 选择：

- `nullcontext`
- 或 `torch.cuda.amp.autocast`

#### 6.5 初始化 wandb/swanlab

如果启用，就建立实验记录。

#### 6.6 初始化三类模型/引擎

- 训练模型 `model`
- 参考模型 `ref_model`
- 奖励模型 `reward_model`
- rollout engine

#### 6.7 构造数据集与 dataloader

- `AgentRLDataset`
- `DistributedSampler`
- `DataLoader`

#### 6.8 构造优化器和学习率调度器

- `AdamW`
- `CosineAnnealingLR`

#### 6.9 可选 resume / compile / DDP

- load state dict
- `torch.compile`
- `DistributedDataParallel`

#### 6.10 进入 epoch 训练循环

- 设置 sampler epoch
- 处理 skip 恢复
- 调 `rl_train_epoch(...)`

#### 6.11 清理分布式进程组

训练结束后 `dist.destroy_process_group()`

### 这一部分的本质

它是在做“训练任务总装配”：  
把模型、数据、优化器、rollout、奖励、日志、分布式这些都接起来，然后开跑。







下面我按 **模块划分 + 行号** 来做“逐行解释”。我会尽量做到两件事：

1. **不漏代码行**
    
2. **每行都解释它在 Agent RL 训练链条里的作用**
    

这份 `train_agent.py` 可以分成 6 个模块：导入与环境准备、工具与 mock 环境、多轮 rollout、奖励计算、RL 训练主循环、`main` 启动逻辑。整体代码从文件头到结尾依次组织成这 6 部分。

---

# 模块 1：导入与环境准备（第 1–32 行）

这一段的作用是：把脚本变成一个可以独立运行的训练入口，并把后面 Agent RL 所需的所有依赖接进来。

### 第 1 行

`import os`

导入操作系统接口库。后面用它来做路径拼接、创建输出目录等，比如 `os.path.abspath(...)`、`os.makedirs(...)`。

### 第 2 行

`import sys`

导入 Python 解释器相关接口。这里主要用于修改 `sys.path`，让脚本能找到项目根目录下的模块。

### 第 3 行

空行。

只是做视觉分隔，没有运行语义。

### 第 4 行

`__package__ = "trainer"`

手动设置当前脚本的包上下文为 `trainer`。  
这样做通常是为了让相对导入或包内组织更稳定，尤其是在“直接运行脚本”而不是“作为包模块运行”时。

### 第 5 行

`sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`

这一行很关键。拆开看：

- `__file__`：当前脚本文件路径
    
- `os.path.dirname(__file__)`：当前脚本所在目录
    
- `os.path.join(..., '..')`：取上一级目录
    
- `os.path.abspath(...)`：变成绝对路径
    
- `sys.path.append(...)`：把这个路径加入模块搜索路径
    

效果是：**把项目根目录加到 Python 搜索路径里**，这样后面才能导入 `models.model_minimind`、`dataset.lm_dataset`、`trainer.trainer_utils` 等项目内模块。

### 第 6 行

空行。

### 第 7 行

`import re`

正则表达式库。后面用于：

- 解析 `<tool_call>...</tool_call>`
    
- 做数字抽取
    
- 做 reward 中的文本模式匹配
    

### 第 8 行

`import gc`

垃圾回收库。  
这份代码里实际上 **导入了但没显式使用**。通常是作者预留给显存/内存回收调试时用的。

### 第 9 行

`import json`

JSON 编解码。后面工具调用参数和工具结果都大量依赖 JSON。

### 第 10 行

`import math`

数学库。后面：

- mock 数学计算工具里会用
    
- 学习率步数计算也会用 `math.ceil`
    

### 第 11 行

`import random`

随机库。后面用来决定本次 rollout 是否打开 thinking：

```python
open_thinking = random.random() < thinking_ratio
```

### 第 12 行

`import signal`

信号机制库。后面在 `execute_tool` 里用 `signal.alarm(1)` 给工具执行加 1 秒超时保护。

### 第 13 行

`import argparse`

命令行参数解析库。后面的 `main` 几乎所有训练配置都通过它读入。

### 第 14 行

`import warnings`

Python 警告处理。后面直接把警告关掉。

### 第 15 行

`import torch`

PyTorch 主库。整个训练脚本的核心依赖。

### 第 16 行

`import torch.nn.functional as F`

导入函数式接口，后面会用：

- `F.log_softmax(...)`
    
- `gather(...)` 取 token 对应 logprob
    

### 第 17 行

`import torch.distributed as dist`

分布式训练接口。后面用于：

- 初始化 rank/device
    
- 判断是否启用 DDP
    
- 销毁进程组
    

### 第 18 行

`from contextlib import nullcontext`

导入“空上下文管理器”。  
后面在 CPU 模式下替代 `autocast` 使用：

- CUDA 上：`torch.cuda.amp.autocast(...)`
    
- CPU 上：`nullcontext()`
    

这样可以统一写成 `with autocast_ctx:`。

### 第 19 行

`from torch import optim`

导入优化器模块，后面会构造 `optim.AdamW(...)`。

### 第 20 行

`from torch.nn.parallel import DistributedDataParallel`

导入 DDP 封装器，多卡训练时把模型包起来。

### 第 21 行

`from torch.utils.data import DataLoader, DistributedSampler`

导入数据加载器和分布式采样器：

- `DataLoader`：按 batch 取数据
    
- `DistributedSampler`：多卡下把数据分给不同进程
    

### 第 22 行

`from torch.optim.lr_scheduler import CosineAnnealingLR`

导入余弦退火学习率调度器。后面训练中每步更新学习率。

### 第 23 行

`from transformers import AutoTokenizer`

导入 HuggingFace 的自动 tokenizer 类。  
这份脚本里 **导入了但没直接用**，因为真正初始化 tokenizer 用的是 `init_model(...)`。

### 第 24 行

`from models.model_minimind import MiniMindConfig, MiniMindForCausalLM`

导入 MiniMind 的配置类和模型类。

- `MiniMindConfig`：定义 hidden_size、层数、max_seq_len、是否 MoE 等
    
- `MiniMindForCausalLM`：真正的因果语言模型
    

这说明 `train_agent.py` 本身不定义模型结构，只负责训练逻辑。

### 第 25 行

`from dataset.lm_dataset import AgentRLDataset`

导入 Agent 强化学习专用数据集。  
说明这个脚本吃的数据不是普通文本，而是包含：

- `messages`
    
- `tools`
    
- `gt`
    

的 agent 任务样本。

### 第 26 行

`from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel`

这一行导入了一组训练公共工具：

- `Logger`：统一日志打印
    
- `is_main_process`：判断主进程
    
- `lm_checkpoint`：保存/恢复 checkpoint
    
- `init_distributed_mode`：初始化分布式
    
- `setup_seed`：设随机种子
    
- `SkipBatchSampler`：恢复训练时跳过已跑 batch
    
- `init_model`：模型和 tokenizer 初始化
    
- `LMForRewardModel`：奖励模型包装器
    

可见这个项目把很多训练公共逻辑抽到 `trainer_utils.py` 里了。

### 第 27 行

`from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps`

导入 rollout 引擎相关接口：

- `create_rollout_engine(...)`：创建 rollout 执行器
    
- `compute_per_token_logps(...)`：计算参考模型每 token 的 log probability
    

这两者是 RL 训练的关键依赖。

### 第 28 行

空行。

### 第 29 行

`warnings.filterwarnings('ignore')`

关闭警告输出。  
作用是让训练日志更干净，但代价是某些潜在问题提示也会被吞掉。

### 第 30 行

空行。

### 第 31 行

注释：  
`# ================================ 工具与 Reward = Start ================================`

仅作为大模块边界标记。说明从这里开始进入“工具和奖励逻辑”。

### 第 32 行

空行。

---

# 模块 2：工具与 mock 环境（第 33–96 行）

这一段代码定义了 agent 可调用工具、mock 工具执行器、参数校验，以及重复惩罚函数。它相当于 RL 训练里的一个“简化环境”。

## 2.1 `rep_penalty`：重复惩罚（第 33–36 行）

### 第 33 行

`def rep_penalty(text, n=3, cap=0.5):`

定义重复惩罚函数。

- `text`：待检查文本
    
- `n=3`：默认看 3-gram 重复
    
- `cap=0.5`：惩罚上限 0.5
    

这个函数后面会在 reward 里扣分，防止模型啰嗦、循环复读。

### 第 34 行

`toks = re.findall(r"\w+|[^\w\s]", text.lower())`

把文本先转小写，再切成 token。  
这个正则意思是：

- 匹配单词字符序列 `\w+`
    
- 或匹配非单词非空白字符 `[^\w\s]`，也就是标点等
    

例如 `"Hello, world!"` 会被拆成类似 `["hello", ",", "world", "!"]`。

### 第 35 行

`grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]`

构造连续的 n-gram 列表。  
默认 `n=3` 时，就是所有连续三元组。  
例如 token 序列 `[a,b,c,d]` 会得到 `[(a,b,c),(b,c,d)]`。

### 第 36 行

`return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0`

计算重复惩罚值：

- `len(grams)`：总 n-gram 数
    
- `len(set(grams))`：去重后的 n-gram 数
    
- 两者差值越大，表示重复越严重
    
- 最后乘一个比例系数，再用 `min(cap, ...)` 限幅到 `cap`
    

如果文本太短，没有任何 n-gram，则返回 0。

---

## 2.2 `TOOLS`：工具 schema 定义（第 38–46 行）

### 第 38 行

注释：  
`# ======== 工具定义 ========`

表示下面开始定义 Agent 可见工具。

### 第 39 行

`TOOLS = [`

定义工具列表。每个元素都是 OpenAI function calling 风格的 schema 字典。

### 第 40 行

定义工具 `calculate_math`：

- 名字：`calculate_math`
    
- 描述：计算数学表达式
    
- 参数：一个对象，其中必须有 `expression: string`
    

这告诉模型：如果要算式子，应构造这样的函数调用。

### 第 41 行

定义工具 `unit_converter`：

- 需要 `value`
    
- 需要 `from_unit`
    
- 需要 `to_unit`
    

典型用途是单位换算。

### 第 42 行

定义工具 `get_current_weather`：

- 参数只要 `location`
    

表示天气查询工具。

### 第 43 行

定义工具 `get_current_time`：

- 参数 `timezone`
    
- 默认 `Asia/Shanghai`
    
- `required` 为空，表示不传也可
    

说明这个工具可无参调用。

### 第 44 行

定义工具 `get_exchange_rate`：

- `from_currency`
    
- `to_currency`
    

表示汇率查询。

### 第 45 行

定义工具 `translate_text`：

- `text`
    
- `target_language`
    

表示翻译工具。

### 第 46 行

`]`

结束工具列表定义。

---

## 2.3 mock 数据表（第 48–53 行）

这些不是在线 API，而是训练用的“伪环境数据”。

### 第 48 行

注释：  
`# ======== 模拟数据 ========`

说明下面是 mock world knowledge。

### 第 49 行

`WEATHER_DATA = {...}`

定义城市到 `(温度, 天气描述)` 的映射。  
例如：

- 北京 → `("28°C", "晴")`
    
- London → `("5°C", "小雨")`
    

模型调用天气工具时，最后就是从这里取值。

### 第 50 行

`TIME_DATA = {...}`

定义时区到固定时间字符串的映射。  
注意这里是**固定时间**，不是实时系统时间，说明训练环境追求的是稳定可复现。

### 第 51 行

`EXCHANGE_DATA = {...}`

定义部分货币对的固定汇率。  
也是 mock 查询源。

### 第 52 行

`TRANSLATE_DATA = {...}`

定义少量固定翻译对。  
训练时翻译工具调用返回就从这里查。

### 第 53 行

`UNIT_DATA = {...}`

定义单位换算倍率。  
例如：

- `km_miles`
    
- `kg_pounds`
    
- `celsius_fahrenheit`
    

等。

---

## 2.4 `MOCK_RESULTS`：模拟执行器（第 55–63 行）

### 第 55 行

注释：  
`# ======== 模拟执行 ========`

说明下面把“工具名”映射到“执行函数”。

### 第 56 行

`MOCK_RESULTS = {`

定义一个字典：工具名 → 执行 lambda。

### 第 57 行

`"calculate_math": lambda args: ...`

数学工具执行逻辑。内部做了几步：

- `args.get("expression", "0")`：取表达式，默认 `"0"`
    
- 把 `^` 替换为 `**`
    
- 把中文括号、乘除号等替换成 Python 可执行形式
    
- 用 `eval(...)` 在受限环境里执行
    
- 返回 `{"result": ...}`
    

这相当于一个简化计算器。  
注意它通过 `{"__builtins__": {}, "math": math}` 限制了可访问对象，但 `eval` 依旧是敏感点，只是这里用于训练 mock。

### 第 58 行

`"unit_converter": lambda args: ...`

单位换算逻辑：

- 读取 `value`
    
- 拼出 key，如 `"km_miles"`
    
- 去 `UNIT_DATA` 里查倍率
    
- 相乘并四舍五入到 4 位
    

返回 `{"result": ...}`。

### 第 59 行

`"get_current_weather": lambda args: ...`

天气工具逻辑：

- 取 `location`
    
- 从 `WEATHER_DATA` 查 `(温度, 天气)`
    
- 如果没有，默认 `("22°C", "晴")`
    
- 组装成带 `city / temperature / humidity / condition` 的字典
    

这里湿度固定 `"65%"`，也是 mock 常量。

### 第 60 行

`"get_current_time": lambda args: ...`

时间工具逻辑：

- 按 `timezone` 去 `TIME_DATA` 查
    
- 查不到默认上海时间
    
- 返回 `{"datetime": ..., "timezone": ...}`
    

### 第 61 行

`"get_exchange_rate": lambda args: ...`

汇率工具逻辑：

- 从 `args` 里取出源货币和目标货币
    
- 用二元组 `(from, to)` 查 `EXCHANGE_DATA`
    
- 查不到默认 1.0
    

返回 `{"from": ..., "to": ..., "rate": ...}`。

### 第 62 行

`"translate_text": lambda args: ...`

翻译工具逻辑：

- 用 `(text, target_language)` 查 `TRANSLATE_DATA`
    
- 查不到就原样返回原文本
    

作用是保证工具调用永远有一个可返回值。

### 第 63 行

`}`

结束 `MOCK_RESULTS` 定义。

---

## 2.5 `CHECK_ARGS`：参数校验器（第 65–73 行）

### 第 65 行

注释：  
`# ======== 参数校验 ========`

表示下面定义每个工具的“参数是否合法”。

### 第 66 行

`CHECK_ARGS = {`

建立工具名 → 参数检查函数 的字典。

### 第 67 行

`"calculate_math": lambda a: bool(a.get("expression")),`

只要存在非空 `expression` 就算合法。

### 第 68 行

`"unit_converter": lambda a: ...`

要求：

- `value is not None`
    
- `from_unit` 存在
    
- `to_unit` 存在
    

### 第 69 行

`"get_current_weather": lambda a: bool(a.get("location")),`

天气工具要求 location 非空。

### 第 70 行

`"get_current_time": lambda a: True,`

时间工具无论传没传都算合法。

### 第 71 行

`"get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),`

要求源货币和目标货币都存在。

### 第 72 行

`"translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),`

要求待翻译文本和目标语言都存在。

### 第 73 行

`}`

结束校验器定义。

---

## 2.6 工具调用解析与执行（第 75–94 行）

### 第 75 行

注释：  
`# ======== 工具调用解析与执行 ========`

表示开始写“从模型输出里解析工具调用”和“执行工具”。

### 第 76 行

`def parse_tool_calls(text):`

定义一个函数，从模型输出文本里解析出工具调用列表。

### 第 77 行

`calls = []`

初始化结果列表。

### 第 78 行

`for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):`

用正则找出所有 `<tool_call>...</tool_call>` 包裹的内容。

- `re.DOTALL` 让 `.` 也匹配换行
    
- 因此工具调用内容可以跨行
    

这说明模型工具调用输出协议是：**把 JSON 放在 `<tool_call>` 标签里**。

### 第 79 行

`try: calls.append(json.loads(m.strip()))`

尝试把标签内字符串解析成 JSON 字典，并加入列表。

### 第 80 行

`except: pass`

如果 JSON 解析失败，直接忽略这一段。  
说明这里容忍模型输出格式不稳定。

### 第 81 行

`return calls`

返回所有成功解析出来的工具调用。

### 第 82 行

空行。

### 第 83 行

`def execute_tool(name, args):`

定义工具执行入口。输入是工具名和参数。

### 第 84 行

`fn = MOCK_RESULTS.get(name)`

从 mock 执行表里取出对应工具函数。

### 第 85 行

`if not fn: return None`

如果工具名不存在，返回 `None`。  
这在后面会被转成错误 JSON。

### 第 86 行

`try:`

进入异常保护块。

### 第 87 行

`signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))`

设置闹钟信号处理函数：一旦超时，就抛 `TimeoutError`。  
这是一种比较“硬”的超时机制。

### 第 88 行

`signal.alarm(1)`

设置 1 秒后触发超时。  
即工具执行最多只允许 1 秒。

### 第 89 行

`return fn(args)`

真正执行工具函数并返回结果。

### 第 90 行

`except:`

捕获执行期异常。包括超时、参数错误、计算错误等。

### 第 91 行

`return None`

一旦执行失败，统一返回 `None`。

### 第 92 行

`finally:`

进入 finally，确保闹钟被清理。

### 第 93 行

`try: signal.alarm(0)`

取消闹钟。避免影响后续逻辑。

### 第 94 行

`except: pass`

取消闹钟若失败也忽略。

### 第 95–96 行

空行 + 下一模块注释。  
表示下面进入 rollout。

---

# 模块 3：多轮 Rollout（第 97–179 行）

这一部分是 Agent RL 的“采样轨迹收集器”。它不直接训练模型，而是让当前策略去生成回答、触发工具、观察工具结果，并把整条轨迹收集回来。

## 3.1 `rollout_single`：单样本多轮交互（第 97–156 行）

### 第 97 行

`def rollout_single(...):`

定义单条样本的 rollout 过程。  
输入包括：

- `rollout_engine`
    
- `tokenizer`
    
- `messages`
    
- `tools`
    
- 最大轮数
    
- 单轮最大生成长度
    
- 是否开启 thinking 的概率
    
- device
    

输出是一条完整轨迹的各种信息。

### 第 98 行

`all_outputs = []`

保存每一轮 assistant 输出文本。

### 第 99 行

`prompt_ids = None`

保存“最初始 prompt”的 token id。  
只在第一轮构造一次。

### 第 100 行

`response_ids = []`

保存后续所有 response 相关 token id。  
注意这里不仅包括模型直接生成 token，也会包括后面观察到的 tool 消息 token。

### 第 101 行

`response_mask = []`

保存与 `response_ids` 对应的 mask：

- 1：这是模型生成的 token，应参与 RL loss
    
- 0：这是工具观测追加的 token，不应参与策略更新
    

这是整个脚本一个很关键的设计点。

### 第 102 行

`response_old_logps = []`

保存 rollout 时策略模型给这些生成 token 的旧 logprob。  
后面 PPO/GRPO/CISPO 会拿它和当前模型重新算的 logprob 比值。

### 第 103 行

`final_context = ""`

保存最后形成的完整上下文文本。

### 第 104 行

`unfinished = False`

标记是否在达到 `max_turns` 时仍未完成。

### 第 105 行

`open_thinking = random.random() < thinking_ratio`

按概率决定这条 rollout 是否开启 thinking 模式。  
这说明训练时并不是每条样本都要求思维链，而是做了随机混合。

### 第 106 行

`for turn in range(max_turns):`

开始多轮循环，最多交互 `max_turns` 轮。

### 第 107 行

`context = tokenizer.apply_chat_template(...)`

把当前 `messages + tools + thinking 开关` 渲染成一段聊天模板文本，并加上 generation prompt。  
这是送给模型生成的真实上下文字符串。

### 第 108 行

`inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)`

把文本 tokenize 成张量并移到指定设备。  
`add_special_tokens=False` 说明聊天模板本身已包含所需格式，不再额外加 special tokens。

### 第 109 行

`context_ids = inputs["input_ids"][0].tolist()`

取出当前上下文 token id 列表。因为这里只 rollout 单条样本，batch 维是 1。

### 第 110–111 行

```python
if prompt_ids is None:
    prompt_ids = context_ids
```

只在第一轮时，把当前上下文记为原始 prompt。  
后面即使上下文因工具结果变长，也不再改 `prompt_ids`。

### 第 112–118 行

调用 rollout 引擎：

- `prompt_ids=inputs["input_ids"]`
    
- `attention_mask=inputs["attention_mask"]`
    
- `num_generations=1`
    
- `max_new_tokens=max_new_tokens`
    
- `temperature=0.8`
    

意思是：对这条上下文采样 1 个回答。  
真正的采样实现藏在 `rollout_engine.rollout(...)` 里。

### 第 119 行

`new_ids = rollout_result.completion_ids[0].tolist()`

取出新生成 completion 的 token ids。

### 第 120 行

`new_logps = rollout_result.per_token_logps[0].tolist()`

取出这些生成 token 对应的逐 token log probability。  
这就是“old policy logps”。

### 第 121 行

检查 token 数和 logprob 数是否一致，不一致则打日志。  
说明作者默认这两者应一一对应。

### 第 122 行

`pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]`

把 `pad` 和 `eos` token 过滤掉。  
即真正保留下来参与轨迹统计的，是有效生成 token。

### 第 123 行

`new_ids = [t for t, _ in pairs]`

保留过滤后的 token ids。

### 第 124 行

`new_logps = [lp for _, lp in pairs]`

保留对应过滤后的 logprobs。

### 第 125 行

`new_text = rollout_result.completions[0]`

取得本轮生成的文本字符串。

### 第 126 行

`all_outputs.append(new_text)`

把这一轮输出记到历史里。

### 第 127 行

`response_ids.extend(new_ids)`

把本轮生成 token 并入总 response token 列表。

### 第 128 行

`response_mask.extend([1] * len(new_ids))`

这些 token 都是模型主动生成的，所以 mask 记 1。

### 第 129 行

`response_old_logps.extend(new_logps)`

把对应旧 logprob 也并入。

### 第 130 行

`final_context = context + new_text`

先暂时把“上下文 + 本轮输出”看成当前最终上下文。

### 第 131 行

`calls = parse_tool_calls(new_text)`

从本轮输出里解析工具调用。

### 第 132–133 行

```python
if not calls:
    break
```

如果这轮没有工具调用，说明 agent 认为可以直接结束。  
于是多轮交互就停止。

### 第 134 行

`unfinished = turn == max_turns - 1`

如果当前正好已经是最后一轮，但还在调工具，则记为未完成。  
这会在 reward 阶段扣分。

### 第 135 行

`messages.append({"role": "assistant", "content": new_text})`

把本轮 assistant 输出追加回消息历史。  
这样下一轮渲染聊天模板时，模型能看到自己刚说过什么。

### 第 136 行

`for call in calls:`

遍历本轮解析出的每个工具调用。

### 第 137 行

`name, raw = call.get("name", ""), call.get("arguments", {})`

取工具名和参数。参数可能已经是 dict，也可能还是字符串。

### 第 138–140 行

如果 `raw` 是字符串，则尝试 `json.loads(raw)` 转成字典；失败则用空字典。  
这是在兼容模型不同的输出习惯。

### 第 141 行

`result = execute_tool(name, raw)`

真正执行 mock 工具。

### 第 142 行

`result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]`

把工具结果转成字符串：

- 有结果：序列化成 JSON
    
- 没结果：给一个错误 JSON
    
- 最后截断到 2048 个字符
    

作者特别写了注释，防止工具结果太大把 tokenizer 撑爆。

### 第 143 行

`messages.append({"role": "tool", "content": result_str})`

把工具返回结果作为 `tool` 角色消息加入对话历史。  
这就是 Agent 的“观察”步骤。

### 第 145 行

`observe_context = tokenizer.apply_chat_template(...)`

工具结果追加后，再重新渲染一次聊天模板。  
这里 `add_generation_prompt=not unfinished`：

- 若没到上限，会继续给下一轮生成提示
    
- 若已到上限，就不再加 generation prompt
    

### 第 146 行

`observe_ids = tokenizer(... )["input_ids"][0].tolist()`

把“观察后的完整上下文”也 token 化，拿到 token ids。

### 第 147 行

`current_len = len(prompt_ids) + len(response_ids)`

计算当前已经记录过的总长度：

- 原始 prompt 长度
    
- 已保存 response token 长度
    

### 第 148 行

`obs_delta = observe_ids[current_len:]`

取出“新多出来”的那一段 token。  
这段就是工具返回结果以及可能新增模板标记造成的增量。

### 第 149 行

`response_ids.extend(obs_delta)`

把这些观察 token 也拼进 response 序列。  
这样后面训练时，输入序列是完整的“prompt + model输出 + tool观察 + 后续输出 ...”链条。

### 第 150 行

`response_mask.extend([0] * len(obs_delta))`

但是这些 token 不是策略生成动作，而是环境反馈，所以 mask 记 0，不参与策略梯度。  
这是这份代码里最重要的一个“动作/观察分离”机制。

### 第 151 行

`response_old_logps.extend([0.0] * len(obs_delta))`

观察 token 没有 old policy logprob，因此用 0.0 占位。

### 第 152 行

`final_context = observe_context`

把最终上下文更新成加入工具观察后的版本。

### 第 154 行

`final_output = all_outputs[-1] if all_outputs else ""`

取最后一轮 assistant 输出作为最终 completion。

### 第 155 行

`prompt_ids = prompt_ids or []`

防御性写法。确保返回时不是 `None`。

### 第 156 行

`return ...`

返回 8 个东西：

1. `final_output`
    
2. `final_context`
    
3. `prompt_ids`
    
4. `response_ids`
    
5. `response_mask`
    
6. `response_old_logps`
    
7. `list(all_outputs)`：每轮输出文本
    
8. `unfinished`
    

这些正是后面 reward 和 RL loss 要用的全部材料。

---

## 3.2 `rollout_batch`：批量采样（第 158–179 行）

### 第 158 行

`def rollout_batch(...):`

定义 batch 级 rollout。  
它本质上是对 `rollout_single` 的外层包装。

### 第 159–166 行

初始化 8 个列表，分别批量收集：

- completions
    
- contexts
    
- prompt ids
    
- response ids
    
- response masks
    
- old logps
    
- turn outputs
    
- unfinished flags
    

它们和 `rollout_single` 返回值一一对应，只是从单条变成批量。

### 第 167 行

`for messages, tools in zip(messages_batch, tools_batch):`

遍历 batch 中每个样本的：

- 对话消息
    
- 可用工具
    

### 第 168 行

`for _ in range(num_gen):`

同一个 prompt 采样 `num_gen` 个候选。  
这一步对 GRPO/CISPO 很关键，因为后面 reward 是按组归一化的。

### 第 169 行

`msgs_copy = [dict(m) for m in messages]`

拷贝消息列表，避免 `rollout_single` 在多轮交互中修改原始 batch 数据。

### 第 170 行

调用 `rollout_single(...)`，得到这一条采样轨迹的全部结果。

### 第 171–178 行

把 `rollout_single` 返回的各类结果分别追加到批量列表中。  
这里等价于把维度从：

- 样本维
    
- 采样维
    

展平成一个总列表。

### 第 179 行

返回 8 个批量列表。  
后续训练把它们当作“已采样好的经验池”。

---

# 模块 4：奖励计算（第 182–240 行）

这一部分定义了“什么样的轨迹是好的”。在 RL 里，这部分几乎决定了模型最终会被塑造成什么风格。

## 4.1 `validate_gt_in_text`（第 182–185 行）

### 第 182 行

`def validate_gt_in_text(text, gt_list):`

定义一个辅助函数：检查最终文本里是否命中了 ground truth 列表中的项目。

### 第 183 行

`text, text_num = str(text), str(text).replace(',', '')`

做两份文本：

- `text`：原文本字符串
    
- `text_num`：去掉逗号的文本，方便数字比较，如 `1,000` → `1000`
    

### 第 184 行

`nums = [float(x) for x in re.findall(...)]`

从文本里抽取所有数字，转成 float 列表。  
正则支持：

- 正负号
    
- 整数
    
- 小数
    

### 第 185 行

返回一个集合推导式，含义是：

对 `gt_list` 中每个 `g`，如果满足以下任一条件，就认为命中：

1. `g` 的字符串形式（去空格后）出现在文本里，忽略大小写
    
2. `g` 本身是数值串，并且文本里抽取出的某个数字与它数值相等（误差 `< 1e-6`）
    

这样就能同时兼容：

- 文本答案
    
- 数值答案
    

---

## 4.2 `calculate_rewards`（第 187–238 行）

### 第 187 行

定义奖励函数主入口。  
输入包括：

- `prompts`
    
- `completions`
    
- `gt_batch`
    
- `tools_batch`
    
- `num_gen`
    
- `reward_model`
    
- `turn_outputs_batch`
    
- `unfinished_batch`
    

输出是一个 reward tensor。

### 第 188 行

`rewards = torch.zeros(len(completions), device=device)`

为每条 completion 初始化一个 reward 槽位。

### 第 189 行

`for idx, response in enumerate(completions):`

逐个 completion 打分。

### 第 190 行

`reward, answer = 0.0, response`

初始化该样本 reward 为 0，并先把 `answer` 设成整个 response。

### 第 191 行

`sample_idx = idx // num_gen`

把“第几个 completion”映射回“原始第几个 prompt”。  
因为每个 prompt 会采样 `num_gen` 个 completion。

### 第 192 行

`tools = tools_batch[sample_idx]`

取这个 prompt 对应的合法工具集合。

### 第 193 行

`turn_outputs = ...`

如果传入了多轮输出历史，就用它；否则退化为单轮 response 列表。

### 第 194 行

`unfinished = ...`

取该轨迹是否未完成。

### 第 195 行

`turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]`

对每一轮输出做后处理：

- 如果包含 `</think>`，只取思考结束后的回答部分
    
- 否则取整段去空格文本
    

也就是说，后面很多 reward 更关注“显式回答部分”，而不是思考过程本身。

### 第 196 行

`answer = turn_answers[-1] if turn_answers else response.strip()`

把最后一轮回答视为最终答案。

### 第 197 行

`valid_names = {t['function']['name'] for t in tools} if tools else set()`

抽出合法工具名集合。  
后面判断模型是否调用了允许的工具。

### 第 198 行

`tool_calls = []`

初始化工具调用列表。

### 第 199 行

遍历每轮回答，用 `parse_tool_calls` 解析所有工具调用并合并。  
因此 reward 看的是整条轨迹，不只最后一轮。

### 第 200 行

对 tool 标签不匹配做惩罚：

- `<tool_call>` 数量
    
- `</tool_call>` 数量
    

如果不闭合或数量不一致，按差值扣分。  
这在鼓励模型遵守输出协议。

---

### 无工具调用分支（第 201–217 行）

### 第 202 行

`if not tool_calls:`

若整条轨迹没有调任何工具，走“纯文本回答”打分逻辑。

### 第 203 行

根据回答长度加/扣分：

- 5 到 800 字符：+0.5
    
- 否则：-0.5
    

这是一个很粗糙的长度先验，防止过短或过长。

### 第 204 行

`if '</think>' in response:`

如果有显式 thinking 结构，则额外检查思考段质量。

### 第 205 行

`think, answer = response.split('</think>', 1)`

把响应分成：

- 思考部分
    
- 最终回答部分
    

只在第一次 `</think>` 处分割。

### 第 206 行

思考长度奖励：

- 20 到 300 字符：+1.0
    
- 否则：-0.5
    

说明作者希望 thinking 不要太短也别太长。

### 第 207 行

闭合奖励：

- 恰好出现一个 `</think>`：+0.25
    
- 否则：-0.25
    

鼓励严格的结构输出。

### 第 208 行

`answer = answer.strip()`

去掉回答段前后空白。

### 第 209 行

`if reward_model is not None:`

如果配置了奖励模型，则引入 RM 分数。

### 第 210 行

`prompt = prompts[sample_idx]`

取该样本的原始 prompt 文本。

### 第 211 行

定义一个 regex，用来从聊天模板中提取：

- `system`
    
- `user`
    
- `assistant`
    

消息块。  
模板格式看起来是 `<|im_start|>role ... <|im_end|>`。

### 第 212 行

`matches = re.findall(pattern, prompt, re.DOTALL)`

从 prompt 中抽取所有消息块。

### 第 213 行

把正则结果转成 reward model 所需的 `messages` 结构。

### 第 214 行

`score = reward_model.get_score(messages, answer)`

调用奖励模型给最终回答打分。  
这相当于“偏好模型”或“response quality model”。

### 第 215 行

`reward += score`

把奖励模型分数并入总 reward。

### 第 216 行

`reward -= rep_penalty(answer)`

减去重复惩罚。

### 第 217 行

`rewards[idx] = max(min(reward, 3.0), -3.0)`

把 reward clip 到 `[-3, 3]`。  
避免极端 reward 破坏训练稳定性。

---

### 有工具调用分支（第 218–237 行）

### 第 219 行

`else:`

若调用了工具，则走“工具使用型 agent”打分逻辑。

### 第 220 行

`gt = gt_batch[sample_idx]`

取该样本的 ground truth 列表。  
通常是工具结果中应该出现的关键信息。

### 第 221 行

`valid_call_count = 0`

统计“合法工具调用”的数量。

### 第 222 行

遍历所有工具调用。

### 第 223 行

取每次调用的：

- `name`
    
- `arguments`
    

### 第 224–226 行

如果参数还是字符串，就尝试反序列化成字典；失败则置为空字典。

### 第 227 行

`check = CHECK_ARGS.get(name)`

取该工具对应的参数校验函数。

### 第 228 行

`valid_call_count += int(bool(name in valid_names and check and check(raw)))`

若满足以下条件，则记一次合法调用：

- 工具名在允许集合中
    
- 存在校验函数
    
- 参数校验通过
    

### 第 229 行

`tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)`

计算工具调用偏差：

- 合法调用数与 GT 长度的差
    
- 再加上多余无效调用数
    

作者希望工具调用次数和目标需求对齐。

### 第 230 行

若 `tool_gap == 0`，奖励 +0.5；否则每个 gap 扣 0.5。  
即鼓励“工具数对齐”。

### 第 232 行

`final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)`

若轨迹未完成，则最终文本直接视为空。  
否则：

- 若答案里还有工具调用标签，则取最后一个 `</tool_call>` 后面的内容
    
- 否则直接用答案
    

目的是只看“最终自然语言答复”，不把 tool call JSON 本身当答案。

### 第 233 行

`verified = validate_gt_in_text(final_text, gt) if gt else set()`

检查最终答案中命中了多少 GT 项。

### 第 234 行

`if gt: reward += 2.5 * len(verified) / len(gt)`

根据 GT 命中比例给分。  
这一项是这个分支里最核心的“任务完成奖励”。

### 第 235 行

`if unfinished: reward -= 0.5`

未完成轨迹额外扣分。

### 第 236 行

`reward -= rep_penalty(final_text if final_text else answer)`

对最终答案再减去重复惩罚。  
若 `final_text` 为空，就退回用 `answer`。

### 第 237 行

再次 clip 到 `[-3, 3]` 后写入 rewards。

### 第 238 行

`return rewards`

返回整批 completion 的 reward tensor。

---

# 模块 5：RL 训练主循环 `rl_train_epoch`（第 241–370 行）

这是全文件最核心的部分：把 rollout 轨迹、reward、old/new/ref logps 拼起来，形成真正的强化学习更新。

> 一个重要观察：这个函数显式参数不多，但实际上读取了很多**外部全局变量**，例如 `tokenizer`、`args`、`model`、`optimizer`、`scheduler`、`lm_config`、`autocast_ctx`。  
> 这说明它虽然写成函数，实际仍高度依赖 `main` 中初始化好的运行环境。

## 5.1 循环开始与 rollout（第 241–251 行）

### 第 241 行

定义 epoch 训练函数。  
参数里有：

- 当前 epoch
    
- dataloader
    
- 总迭代数
    
- rollout engine
    
- reference model
    
- reward model
    
- 恢复训练时起始 step
    
- wandb 句柄
    
- `use_sglang` 标记
    

不过注意：`use_sglang` 参数在函数体中其实没有使用。

### 第 242 行

`last_step = start_step`

记录最后一个实际跑到的 step，用于函数尾部处理“梯度累积没凑满”的情况。

### 第 243 行

`for step, batch in enumerate(loader, start=start_step + 1):`

遍历 dataloader。  
如果是 resume 训练，会从 `start_step + 1` 开始记 step。

### 第 244–246 行

从 batch 中取出三项：

- `messages_batch`
    
- `tools_batch`
    
- `gt_batch`
    

这也印证了 `AgentRLDataset`/`collate_fn` 的输出结构。

### 第 247 行

`last_step = step`

更新最后 step 记录。

### 第 249 行

`with torch.no_grad():`

在 rollout 阶段不需要训练梯度，因此关闭 autograd。  
这能节省显存，也符合“采样轨迹”阶段的需求。

### 第 250 行

调用 `rollout_batch(...)`，拿到 8 组结果：

- completions
    
- contexts
    
- prompt_ids_batch
    
- response_ids_batch
    
- response_masks_batch
    
- response_old_logps_batch
    
- turn_outputs_batch
    
- unfinished_batch
    

这一步就是“用当前策略收集经验”。

## 5.2 把 rollout 样本打包成训练输入（第 252–269 行）

### 第 252 行

重新根据原始 `messages + tools` 渲染每个 prompt 文本。  
这是给 reward model 和调试打印等用途保留原始 prompt。

### 第 253 行

`packed_samples = []`

准备把每个采样样本整理成统一的四元组。

### 第 254 行

遍历每个 rollout 后样本的：

- prompt ids `p`
    
- response ids `r`
    
- response mask `m`
    
- old logps `old_lp`
    

### 第 255 行

`ids = p + r`

完整训练输入序列 = 原始 prompt + 后续所有 response/observation token。

### 第 256 行

`mask = [0] * len(p) + m`

为完整输入构造一个同长 mask：

- prompt 全部是 0
    
- response 部分沿用 rollout 产生的 mask（动作 token 为 1，观察 token 为 0）
    

这就是后面只在“真正动作 token”上算策略损失的关键。

### 第 257 行

`old_logps = [0.0] * max(len(p) - 1, 0) + old_lp`

把 old logprob 对齐到完整序列。  
注意是 `len(p) - 1`，因为语言模型的逐 token logprob 对应的是“预测下一个 token”的位置，长度比 input_ids 少 1。

### 第 258–261 行

如果总长度超过 `args.max_total_len`，就从右侧保留最后一段：

- 截 `ids`
    
- 截 `mask`
    
- 截 `old_logps`
    

也就是说，该训练脚本对过长轨迹采用**保留尾部**策略。  
这是有道理的，因为 Agent 任务的奖励通常更依赖末端决策和最终答案。

### 第 262 行

`prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))`

找到第一个动作 token 的位置。  
如果没有任何 `1`，则 prompt_len 设为整个序列长度。

### 第 263 行

把 `(ids, mask, prompt_len, old_logps)` 打包进列表。

### 第 264 行

`seq_lens = torch.tensor([...], device=args.device)`

记录每个样本真实长度。

### 第 265 行

`max_len = seq_lens.max().item()`

找到本批次最长序列长度。

### 第 266 行

构造 padded `input_ids` 张量。  
短样本在右边补 pad token。

### 第 267 行

把所有 `prompt_len` 拼成张量。  
后面主要用于调试显示 completion 起点。

### 第 268 行

构造 padded `full_response_masks`，类型为 `float32`。  
后面它会和 loss 相乘。

### 第 269 行

构造 padded `old_per_token_logps`。  
长度要对齐到 `max_len - 1`，因为逐 token logprob 对应预测位。

## 5.3 当前策略与参考策略 logprob（第 271–280 行）

### 第 271 行

`model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model`

若模型被 DDP 包裹，则先取出裸模型。  
原因是前向调用和后续某些属性访问更方便。

### 第 272 行

`with autocast_ctx:`

进入自动混合精度上下文。  
在 CUDA 上可能是 bf16/fp16，在 CPU 上是空上下文。

### 第 273 行

`res = model_unwrapped(input_ids)`

前向跑当前策略模型。  
返回对象里至少有：

- `logits`
    
- 可能有 `aux_loss`（MoE 时）
    

### 第 274 行

若启用 MoE，则取 `res.aux_loss`；否则用 0。  
这是 MoE 常见的负载均衡辅助损失。

### 第 275 行

`logits = res.logits[:, :-1, :]`

去掉最后一个时间步。  
原因是第 `t` 个位置的 logits 用来预测第 `t+1` 个 token，所以和 `input_ids[:, 1:]` 对齐。

### 第 276 行

`per_token_logps = F.log_softmax(logits, dim=-1).gather(...).squeeze(-1)`

这一行很关键。它做了：

1. 对 vocab 维做 `log_softmax`，得到每个位置对所有词的对数概率
    
2. 用 `gather` 取出真实下一个 token 的那一项
    
3. `squeeze(-1)` 去掉最后单例维
    

最终得到形状大致为 `[B, T-1]` 的逐 token logprob。  
这是**当前策略**对这条采样轨迹的重新评估结果。

### 第 278–279 行

在 `no_grad` 下，用参考模型计算同一条输入的 `ref_per_token_logps`。  
后面会拿它做 KL 正则，防止策略漂太远。

## 5.4 completion mask、EOS 截断、reward（第 281–290 行）

### 第 281 行

`completion_mask = full_response_masks[:, 1:]`

和 logprob 长度对齐。  
因为 `per_token_logps` 对应的是预测位置，少一位。

### 第 282 行

`is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()`

找出 response 区域中哪些位置是 EOS。

### 第 283 行

先默认每行 EOS 下标都在最后一位。

### 第 284 行

`has_eos = is_eos.any(dim=1)`

判断每个样本是否真的出现过 EOS。

### 第 285 行

对出现过 EOS 的样本，取第一个 EOS 的位置。

### 第 286 行

构造位置索引张量 `[0,1,2,...]`。

### 第 287 行

`completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()`

把 EOS 后面的 token 全部 mask 掉。  
即策略损失只算到第一个 EOS 为止。

### 第 288 行

`token_counts = completion_mask.sum(dim=1)`

统计每个样本有效 completion token 数。

### 第 289 行

`valid_rows = token_counts > 0`

标记哪些样本至少有一个有效 token。  
避免后面除零。

### 第 290 行

调用 `calculate_rewards(...)` 算出每条 completion 的 reward。  
输入里把多轮输出和 unfinished 信息也传进去了。

## 5.5 调试打印（第 292–309 行）

这一段只在 `debug_mode` 打开且达到间隔时执行。作用是把每个 prompt 的每个采样结果完整打印出来。

- 第 293–294 行：打印 GT
    
- 第 296–297 行：定位第 `i` 个样本第 `j` 个生成结果
    
- 第 298 行：取 prompt 长度和序列长度
    
- 第 299–301 行：打印完整 context
    
- 第 302 行：打印长度信息
    
- 第 303–304 行：从 `input_ids` 中解码 completion 片段
    
- 第 305–307 行：打印 completion 文本
    
- 第 308 行：打印 reward
    
- 第 309 行：打印分隔线
    

这块主要用于排查 reward 是否合理、mask 是否对齐、生成内容是否真的符合预期。

## 5.6 advantage 与策略损失（第 311–330 行）

### 第 311 行

`grouped_rewards = rewards.view(-1, args.num_generations)`

把 reward reshape 成 `[样本数, 每样本生成数]`。  
这一步很关键，因为后面是按“同 prompt 的多候选”做组内归一化。

### 第 312 行

计算每组 reward 均值，并 repeat 回展平形状。

### 第 313 行

计算每组 reward 标准差，并 repeat 回展平形状。

### 第 314 行

`advantages = (rewards - mean_r) / (std_r + 1e-4)`

得到组内标准化 advantage。  
这很像 GRPO 思路：不训练 value model，而是用组内相对好坏来构造优势。

### 第 316 行

`kl_div = ref_per_token_logps - per_token_logps`

先取参考策略与当前策略的 logprob 差。

### 第 317 行

`per_token_kl = torch.exp(kl_div) - kl_div - 1`

计算一种逐 token KL 近似/变体。  
该形式常用于对数比值稳定化处理。

### 第 318 行

`ratio = torch.exp(per_token_logps - old_per_token_logps)`

计算当前策略与 rollout 旧策略的概率比值。  
这是 PPO 家族方法的核心量。

### 第 319 行

`if args.loss_type == "cispo":`

根据参数选择两种 loss 之一。

### 第 320 行

`clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()`

对 ratio 只做上界截断，并且 `detach()`。  
这意味着在 CISPO 分支里，比例本身不回传梯度。

### 第 321 行

`per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)`

CISPO 分支的逐 token 损失。  
直觉上是在：

- 用 advantage 调整 token logprob
    
- 再减去 KL 惩罚项
    
- 整体前面加负号，转成最小化目标
    

### 第 323 行

`clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)`

GRPO/PPO 风格的 clip 比值。

### 第 324 行

`per_token_loss1 = ratio * advantages.unsqueeze(1)`

未裁剪的 surrogate。

### 第 325 行

`per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)`

裁剪后的 surrogate。

### 第 326 行

`per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)`

标准 PPO/GRPO 风格：取两者较小值，再加 KL 惩罚，最后取负。

### 第 327–328 行

对逐 token loss 做 mask、按样本归一化、对 batch 求平均：

- 先乘 `completion_mask`
    
- 每个样本除以有效 token 数
    
- 只保留 `valid_rows`
    
- 最后求均值
    

若没有有效行，就返回 0 梯度张量。

### 第 329 行

`loss = (policy_loss + aux_loss) / args.accumulation_steps`

把策略损失和可选 MoE 辅助损失相加，再除以梯度累积步数。  
这是标准梯度累积写法。

### 第 330 行

`loss.backward()`

反向传播累积梯度。

## 5.7 optimizer step、日志、保存（第 332–368 行）

### 第 332 行

`if step % args.accumulation_steps == 0:`

当积满一个 optimizer step 所需的 micro-step 数后，才真正更新参数。

### 第 333 行

如果启用了梯度裁剪，则对模型参数做 `clip_grad_norm_`。

### 第 334 行

`optimizer.step(); scheduler.step(); optimizer.zero_grad()`

一步完成：

1. 更新参数
    
2. 更新学习率
    
3. 清空梯度
    

### 第 335 行

如果是主进程且到了保存间隔，则 `rollout_engine.update_policy(model)`。  
意思是：让 rollout 侧使用最新策略参数。  
如果 rollout 用的是独立引擎（尤其 sglang），这一步就很关键。

### 第 337 行

`if step % args.log_interval == 0 or step == iters:`

达到日志间隔或 epoch 结束时打印指标。

### 第 338 行

`pl = loss.item() * args.accumulation_steps`

把前面除过的 loss 乘回来，得到更接近真实单 step 的 policy loss 数值。

### 第 339 行

`ar = rewards.mean().item()`

平均 reward。

### 第 340 行

`al = token_counts.float().mean().item()`

平均响应长度。

### 第 341 行

计算一个平均 KL 指标。  
这里用的是 `(ref_logps - current_logps)` 在有效 completion token 上的平均。

### 第 342 行

`gs = grouped_rewards.std(...).mean().item()`

每组 reward 标准差的平均值。  
可反映同 prompt 多候选之间的可区分程度。

### 第 343 行

取 advantage 的均值和标准差。

### 第 344 行

取当前学习率。

### 第 345 行

打印完整训练日志。  
包含：

- epoch/step
    
- reward
    
- KL
    
- reward group std
    
- advantage std
    
- loss
    
- avg len
    
- advantage mean
    
- LR
    

### 第 346–347 行

若启用 wandb/swanlab，则把同一批指标也写入可视化平台。

### 第 349 行

`if (step % args.save_interval == 0 or step == iters) and is_main_process():`

在保存间隔或 epoch 末尾，由主进程负责保存模型。

### 第 350 行

`model.eval()`

切到评估模式再保存。  
这是为了让保存时模型处于稳定推理态，避免某些层的训练态副作用。

### 第 351 行

若是 MoE，给权重文件名加 `_moe` 后缀。

### 第 352 行

拼出权重保存路径。文件名包含：

- `save_weight`
    
- `hidden_size`
    
- 可选 `_moe`
    

### 第 353 行

若模型是 DDP，则先取 `model.module`。

### 第 354 行

`raw_model = getattr(raw_model, '_orig_mod', raw_model)`

兼容 `torch.compile` 场景：如果模型被 compile 包裹，则取原始模块。

### 第 355 行

`state_dict = raw_model.state_dict()`

取参数字典。

### 第 356 行

把所有参数转成 half、搬到 CPU，再 `torch.save(...)`。  
这样磁盘占用更小。

### 第 357–358 行

调用 `lm_checkpoint(...)` 保存更完整的 checkpoint，包括：

- model
    
- optimizer
    
- epoch
    
- step
    
- wandb
    
- scheduler
    

这和上面的纯 `.pth` 权重文件不同，它是为 resume 训练服务的。

### 第 359 行

`model.train()`

保存结束后切回训练模式。

### 第 360 行

`del state_dict`

释放本地变量引用。

### 第 362–363 行

删除一些中间张量引用，帮助及时回收显存/内存。

### 第 365 行

处理一个边界情况：如果 epoch 结束时，梯度累积还没凑满一个整步。

### 第 366 行

必要时做梯度裁剪。

### 第 367 行

仍然执行一次：

- `optimizer.step()`
    
- `scheduler.step()`
    
- `optimizer.zero_grad()`
    

避免最后几步梯度白算。

### 第 368 行

若主进程且正好到保存间隔，也更新 rollout policy。

---

# 模块 6：`main` 入口（第 371–487 行）

这一部分负责真正把整套系统装起来并开跑。包括参数解析、分布式初始化、模型/奖励模型/rollout engine 初始化、数据集、优化器、resume、DDP、epoch 循环。

## 6.1 参数定义（第 371–410 行）

### 第 371 行

`if __name__ == "__main__":`

只有直接运行这个脚本时，下面代码才执行。

### 第 372 行

创建参数解析器，任务描述是 `"MiniMind Agent RL"`。

### 第 373–409 行

这一大段全是命令行参数定义。逐个解释：

- 第 373 行 `--save_dir`：权重输出目录
    
- 第 374 行 `--save_weight`：保存权重名基前缀
    
- 第 375 行 `--epochs`：训练轮数
    
- 第 376 行 `--batch_size`：batch 大小
    
- 第 377 行 `--learning_rate`：学习率
    
- 第 378 行 `--device`：默认优先 `cuda:0`，否则 CPU
    
- 第 379 行 `--dtype`：混合精度类型，默认 bf16
    
- 第 380 行 `--num_workers`：DataLoader 线程数
    
- 第 381 行 `--accumulation_steps`：梯度累积步数
    
- 第 382 行 `--grad_clip`：梯度裁剪阈值
    
- 第 383 行 `--log_interval`：日志间隔
    
- 第 384 行 `--save_interval`：保存间隔
    
- 第 385 行 `--hidden_size`：模型隐藏维度
    
- 第 386 行 `--num_hidden_layers`：层数
    
- 第 387 行 `--use_moe`：是否启用 MoE
    
- 第 388 行 `--max_seq_len`：最大上下文长度
    
- 第 389 行 `--max_gen_len`：单次最大生成长度
    
- 第 390 行 `--max_total_len`：训练打包后的最终总长度上界
    
- 第 391 行 `--data_path`：训练数据路径
    
- 第 392 行 `--num_generations`：每个 prompt 采样几个 completion
    
- 第 393 行 `--beta`：KL 正则系数
    
- 第 394 行 `--loss_type`：`grpo` 或 `cispo`
    
- 第 395 行 `--epsilon`：GRPO/PPO clip 范围
    
- 第 396 行 `--epsilon_high`：CISPO ratio 上界
    
- 第 397 行 `--from_weight`：从哪个权重初始化，默认 `full_sft`
    
- 第 398 行 `--from_resume`：是否从 checkpoint 恢复
    
- 第 399 行 `--use_wandb`：是否启用 swanlab/wandb 记录
    
- 第 400 行 `--wandb_project`：项目名
    
- 第 401 行 `--use_compile`：是否 `torch.compile`
    
- 第 402 行 `--debug_mode`：是否打开调试打印
    
- 第 403 行 `--debug_interval`：调试输出间隔
    
- 第 404 行 `--thinking_ratio`：开启 thinking 的概率
    
- 第 405 行 `--reward_model_path`：奖励模型路径
    
- 第 406 行 `--rollout_engine`：rollout 后端，`torch` 或 `sglang`
    
- 第 407 行 `--sglang_base_url`：SGLang 服务地址
    
- 第 408 行 `--sglang_model_path`：SGLang tokenizer 路径
    
- 第 409 行 `--sglang_shared_path`：SGLang 共享存储路径
    

这套参数基本囊括了 Agent RL 训练所需的全部控制面。

### 第 410 行

`args = parser.parse_args()`

解析命令行参数，得到 `args` 对象。

## 6.2 分布式、随机种子、配置（第 412–424 行）

### 第 412 行

`local_rank = init_distributed_mode()`

初始化分布式环境，并返回本进程 local rank。

### 第 413 行

若 `dist` 已初始化，则把 `args.device` 改成 `cuda:{local_rank}`。  
确保每个进程绑定自己的 GPU。

### 第 414 行

`setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))`

设随机种子：

- 基础种子 42
    
- 多卡时每个 rank 加上自己的 rank 值
    

避免不同进程完全相同。

### 第 416 行

`os.makedirs(args.save_dir, exist_ok=True)`

确保输出目录存在。

### 第 417–418 行

构造 `MiniMindConfig`：

- `hidden_size`
    
- `num_hidden_layers`
    
- `max_seq_len=args.max_seq_len + args.max_gen_len`
    
- `use_moe=bool(args.use_moe)`
    

注意这里最大长度不是单纯 `max_seq_len`，而是**上下文长度 + 最长生成长度**。  
因为 rollout 之后的训练输入会把 prompt 和生成一起拼起来。

### 第 419 行

如果 `from_resume == 1`，则从 checkpoint 中读取 `ckp_data`；否则为 `None`。

### 第 421 行

`device_type = "cuda" if "cuda" in args.device else "cpu"`

判断当前运行设备类型。

### 第 422 行

根据参数把 dtype 映射成 PyTorch dtype：

- `"bfloat16"` → `torch.bfloat16`
    
- 否则 → `torch.float16`
    

### 第 423 行

`autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)`

统一构造自动混合精度上下文。  
后面无论 CPU/CUDA，都能统一写 `with autocast_ctx:`。

## 6.3 实验记录、模型初始化（第 425–449 行）

### 第 425 行

`wandb = None`

先初始化为空。

### 第 426 行

只有启用了 `use_wandb` 且当前是主进程，才初始化记录平台。

### 第 427 行

`import swanlab as wandb`

这里虽然变量名叫 `wandb`，实际导入的是 `swanlab`。  
作者是在复用 `wandb` 这个接口名。

### 第 428 行

若是 resume 训练，则尝试从 checkpoint 中恢复 `wandb_id`。

### 第 429 行

有 `wandb_id` 就用 `resume='must'`，否则不设置。

### 第 430 行

调用 `wandb.init(...)` 初始化实验记录。  
名称里包含 epoch、batch_size、learning_rate。

### 第 432 行

`model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)`

初始化训练用策略模型和 tokenizer。  
默认会从 `full_sft` 权重起步。

### 第 434 行

再次初始化一份相同结构、相同初始权重的模型作为 reference model。

### 第 435 行

`ref_model = ref_model.eval().requires_grad_(False)`

把 reference model 设为：

- eval 模式
    
- 不参与梯度更新
    

这是标准 RLHF/GRPO 做法。

### 第 437 行

初始化奖励模型封装器。  
路径由 `reward_model_path` 指定，dtype 固定为 `torch.float16`。

### 第 438 行

打印奖励模型加载成功日志。

### 第 440–449 行

创建 rollout engine。  
传入：

- `engine_type`
    
- 当前策略模型
    
- tokenizer
    
- device
    
- autocast_ctx
    
- sglang 各种配置
    

这说明 rollout 可以有不同后端：

- 本地 torch
    
- 远端/独立的 sglang
    

## 6.4 数据集、优化器、调度器、resume（第 450–466 行）

### 第 450 行

`train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)`

构造训练数据集。  
输入是数据路径、tokenizer 和最大长度。

### 第 451 行

若启用分布式，则构造 `DistributedSampler`，否则为 `None`。

### 第 452 行

`optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)`

用 AdamW 优化训练模型参数。

### 第 453 行

定义一个简洁版 `collate_fn`：

- 汇总 `messages`
    
- 汇总 `tools`
    
- 汇总 `gt`
    

它不做复杂 padding，因为真正 padding 在 rollout 后才做。

### 第 454 行

构造一个只用来统计 `iters` 的 DataLoader。

### 第 455 行

`iters = len(loader_for_count)`

得到每个 epoch 的 batch 数。

### 第 456 行

`total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs`

计算总优化步数。  
注意这里除以的是梯度累积步数，因为不是每个 dataloader step 都会 `optimizer.step()`。

### 第 457 行

构造余弦退火调度器：

- `T_max=total_optimizer_steps`
    
- `eta_min=args.learning_rate / 10`
    

即最终最小 LR 是初始 LR 的 1/10。

### 第 459 行

`start_epoch, start_step = 0, 0`

默认从头开始。

### 第 460 行

如果存在 checkpoint 数据，则恢复。

### 第 461 行

恢复模型参数。

### 第 462 行

恢复优化器状态。

### 第 463 行

恢复调度器状态。

### 第 464 行

恢复起始 epoch。

### 第 465 行

恢复起始 step。

## 6.5 compile、DDP、训练循环（第 467–487 行）

### 第 467 行

若用户要求 compile，则进入该分支。

### 第 468 行

`model = torch.compile(model)`

对模型做 PyTorch 2 的图编译优化。

### 第 469 行

打印 compile 已启用。

### 第 470 行

若分布式已初始化，则进入 DDP 包装分支。

### 第 471 行

`model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}`

告诉 DDP 忽略这两个 buffer/参数。  
这通常与 RoPE 预计算缓存有关，避免 DDP 在它们上面做不必要同步或报错。

### 第 472 行

`model = DistributedDataParallel(model, device_ids=[local_rank])`

把模型包成 DDP。

### 第 473 行

若当前是主进程，则先 `rollout_engine.update_policy(model)`。  
作用是让 rollout 侧加载当前最新策略。

### 第 475 行

开始 epoch 循环，从 `start_epoch` 跑到 `args.epochs - 1`。

### 第 476 行

若有 `train_sampler`，则为本 epoch 设置 epoch 编号。  
这是分布式随机打乱时的标准做法。

### 第 477 行

`setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()`

这行写了两个语句：

1. 重新用 `42 + epoch` 设种子
    
2. 生成一个随机索引序列
    

如果不是分布式，后面会用这个 `indices` 做采样基础。

### 第 478 行

`skip = start_step if (epoch == start_epoch and start_step > 0) else 0`

若当前正是恢复训练的起始 epoch，则需要跳过已经训练过的 step。

### 第 479 行

`batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)`

构造支持“跳过前若干 batch”的采样器。  
这里很巧：

- 分布式时，用 `train_sampler`
    
- 单机时，用随机 `indices`
    

### 第 480 行

用这个 `batch_sampler` 构造真正训练用 dataloader。  
并设置：

- `num_workers`
    
- `pin_memory=True`
    
- `collate_fn`
    

### 第 481 行

如果确实有 skip，要打印说明并走 resume 分支。

### 第 482 行

打印当前 epoch 跳过了多少步。

### 第 483 行

调用 `rl_train_epoch(...)`，注意这里 `iters` 传的是 `len(loader) + skip`，这样日志 step 计数能和恢复前一致。

### 第 484 行

若没有 skip，则走正常训练分支。

### 第 485 行

正常从 step 0 开始调用 `rl_train_epoch(...)`。

### 第 487 行

训练结束后，若启用了分布式，则销毁进程组。

---

# 你读这份文件时，最该抓住的 4 条主线

为了防止“逐行看完还是散的”，我把整份代码再压成 4 条主线：

### 1. 它训练的不是普通 LM，而是会用工具的 Agent

靠的是：

- `TOOLS`
    
- `parse_tool_calls`
    
- `execute_tool`
    
- `rollout_single`
    

这一整套多轮交互机制。

### 2. 它把“模型动作”和“环境观察”分开了

靠的是：

- `response_mask`
    
- 生成 token 记 1
    
- 工具观测 token 记 0
    

这样 loss 只作用在模型真正决策的 token 上。

### 3. 它不是 value-based RL，而是 group-based policy optimization

靠的是：

- 每个 prompt 采样多个 completion
    
- 组内 reward 标准化成 advantage
    
- 用 old/new/ref logps 算 GRPO/CISPO 风格 loss
    

### 4. 它默认站在 `full_sft` 之后

也就是：

**预训练 → SFT → Agent RL**

这个文件属于后训练阶段，不是从零学语言，而是在已有对话模型上继续学“多轮任务执行能力”。

---

# 顺手指出几个你读代码时容易卡住的点

### A. `rl_train_epoch` 为什么能直接用 `tokenizer / args / model / optimizer`？

因为这些变量是在 `main` 里先初始化的，函数体里直接读的是外层全局名字，不是局部参数传入。  
这是一种“脚本式写法”，方便但耦合较强。

### B. 为什么工具观察 token 也会拼进 `input_ids`？

因为模型下一轮生成时必须“看到工具结果”，否则没法继续基于工具返回值作答。  
但这些观察 token 不是模型决策，因此 mask 设 0，不参与策略梯度。

### C. 为什么既要 `model` 又要 `ref_model`？

`model` 是当前被训练的策略；`ref_model` 是固定参考策略，用来提供 KL 正则，限制策略漂移。

### D. 为什么 rollout 里保存 `old_logps`，训练时又重新算 `per_token_logps`？

这是 PPO/GRPO 类方法的标准做法：

- rollout 阶段：记录旧策略概率
    
- 更新阶段：当前模型重新评估同一轨迹
    
- 两者比值形成 `ratio`
    

---

如果你愿意，我下一条可以继续直接做两件事中的一种：  
一是把这份文件画成一张 **“从数据到 loss 的完整流程图”**，二是继续按同样粒度讲 **`rl_train_epoch` 里每个张量的形状怎么变化**。



---
# 整体流程

### 一、`AgentRLDataset` 输出
`_getitem__(index)` 的真实返回值是：

- `messages`: `list[dict]`
- `tools`: `list[dict]` 或 `None`
- `gt:list[dict]`
还不是tensor，这个数据集阶段还没有进行tokenization

RL 和 SFT 最本质的不同：  
**SFT 是先有标准答案，再监督；Agent RL 是先在线生成轨迹，再按结果打分**
体现在代码中：
```
    for step, batch in enumerate(loader, start=start_step + 1):

        messages_batch = batch['messages']

        tools_batch = batch['tools']

        gt_batch = batch['gt']

        last_step = step

  

        with torch.no_grad():

            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)
```


