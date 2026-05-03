MiniMind 里给强化学习阶段准备的“生成执行层 / 推理后端适配层”
**拿当前 policy 模型去生成回答，并把后续 PPO / GRPO 一类算法需要的 rollout 结果统一打包返回**，而且它支持两种后端：  
一类是直接用本地 PyTorch 模型生成，另一类是通过 SGLang 服务走 HTTP 推理。
(同一个训练流程，可以切换不同rollout后端)

在 RLHF / PPO / GRPO 这类流程里，训练不是只看静态标注数据，而是要反复做：

1. 拿当前 policy 模型
2. 对一批 prompt 采样生成多个回答
3. 拿到这些回答对应的 token id
4. 记录这些 token 的 logprob
5. 把这些结果交给 reward / advantage / policy loss 模块

---

## 模块功能

### **1.文件初始化与依赖导入**
```
"""Rollout Engine - 可插拔的推理引擎

python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998

"""

import os

import sys

  

__package__ = "trainer"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

  
#用于发送HTTP请求
import requests

import torch

#抽象基类机制
from abc import ABC, abstractmethod

#用来定义数据类 `RolloutResult`。 让“结果对象”写起来更简洁，不用自己写 `__init__`
from dataclasses import dataclass

from typing import List, Optional, Tuple

from torch import Tensor

from torch.nn.parallel import DistributedDataParallel

from transformers import AutoTokenizer
```
`dataclass`装饰器的方式来定义类，设置默认值很简单，直接在定义属性时就可以设置，不用写`__init__`

---

### **2.`compute_per_token_logps` 工具函数**
给一整段序列 `input_ids`，重新跑模型前向，取出最后 `n_keep` 个 token 的对数概率。是为了精确评估当前 policy 对已经生成出的 token 序列打了多大概率
```
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:

	#如果不需要保留任何token的logprob，直接返回一个空tensor，返回的tensor的shape是[B,0]
    if n_keep <= 0:

        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    per_token_logps = []

    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):

        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row

        per_token_logps.append(

            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)

        )

    return torch.stack(per_token_logps)
```

`input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids`

 `input_ids.is_inference()`
检查这个 tensor 是否是 inference tensor。官方有 `torch.Tensor.is_inference()` 这个接口。

 `detach()`
把它和任何潜在 autograd 历史断开。  
虽然这里的 token id 本来一般就是整型、不参与梯度，但这是一个非常稳妥的“防御式写法”。

 `clone()`
这一步更关键。  
`clone()` 会复制出一份新张量；而这句代码执行时已经**不在** `generate()` 的 inference mode 上下文里了，所以这份新张量是在普通上下文里物化出来的，后续更安全。PyTorch 文档也说明，`torch.tensor(...)` / 复制构造这类操作会创建一个**没有 autograd history 的新 tensor**；`clone()` 在这里承担的是类似“重新物化一份正常张量”的角色。


`torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)`

torch.gather()
torch.gather(input, dim, index, sparse_grad=False, out=None) → Tensor
PyTorch 中用于沿指定维度从输入张量中收集值的函数。它根据索引张量 _index_ 的值，从输入张量 _input_ 中提取对应位置的元素
- 输入 `input = logits_row.log_softmax(dim=-1)`，形状 `[R, V]`
- `dim = 1`，表示沿着第 1 维取值，也就是沿词表维度取值
- `index = ids_row.unsqueeze(1)`，形状 `[R, 1]`

PyTorch 官方文档对 `gather` 的定义是：  
**沿指定维度，根据 index 里的下标去 input 中取值。**

这里的效果就是：

- 第 0 行，从词表维里取出 `ids_row[0]` 对应的 logprob
- 第 1 行，从词表维里取出 `ids_row[1]` 对应的 logprob
- ...
- 第 `R-1` 行也一样

所以你可以把它理解成：

> 从每个时间步的整行词表分布里，抽出“真实生成 token”那个位置的 logprob。

输出形状是：

**`[R, 1]`**



---

### **3.RolloutResult数据结构**
一个**统一结果容器**，统一所有 rollout 输出格式。因为本地引擎和远程引擎内部实现不一样，但是训练器只想拿到统一结果。
这个类就是抽象边界
```
class RolloutResult:
	#完整输出，通常是 `prompt + completion`
    output_ids: Tensor
    
	#只保留模型新生成的部分
    completion_ids: Tensor
    
	#completion 段每个 token 的 logprob
    per_token_logps: Tensor
    
	#把 completion 解码后的文本
    completions: List[str]
```

@dataclass 是Python的数据类装饰器，会自动帮助生成
- `__init__`
- `__repr__`
- 比较方法的一部分

---

### **4.`RolloutEngine` 抽象基类**
定义了一个抽象父类，并声明两个核心接口：
- rollout(...)
- uodate_policy(...)
定义所有 rollout 引擎必须遵守的协议
```
# ===== Rollout 引擎抽象基类 =====

class RolloutEngine(ABC):

    tokenizer = None

    @abstractmethod

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:

        pass

    @abstractmethod

    def update_policy(self, model: torch.nn.Module):

        pass
```


==Notice==：设计模式——==接口抽象+策略模式+简单工厂==
####  A. 第一层：抽象基类 / 接口模式

`RolloutEngine` 先定义了一套协议：

- 必须有 `rollout`
- 必须有 `update_policy`

这就是接口抽象。‘

这个模块是上层模块，抽象接口
经典的 **依赖倒置原则**：

- 上层模块不依赖具体实现
- 上层模块依赖抽象接口

#### B. 第二层：策略模式（Strategy Pattern）
策略模式的定义很简单：

> 把一组可互换的算法/行为封装成不同对象，并通过统一接口使用它们。

在这里，“算法/行为”就是“如何做 rollout”。

有两种策略：

1. `TorchRolloutEngine`
    - 本地模型生成
    - 本地算 logprob
2. `SGLangRolloutEngine`
    - 通过 HTTP 请求远端服务生成
    - 解析返回结果
    - 需要做权重同步

它们内部实现差异很大，但对外都叫：

- `rollout(...)`
- `update_policy(...)`

这就是标准的策略模式。


 **为什么这里特别适合策略模式？**

因为 rollout 后端天然是“可替换”的：

- 本地小实验：用 torch
- 大吞吐推理：用 sglang
- 将来还可以接 vLLM、TGI、TensorRT-LLM

而训练主流程不应该因为后端变化而跟着大改

####  C. 第三层：简单工厂（Simple Factory）

文件最后有：

def create_rollout_engine(...):  
    if engine_type == "torch":  
        return TorchRolloutEngine(...)  
    elif engine_type == "sglang":  
        return SGLangRolloutEngine(...)  
    else:  
        raise ValueError(...)

这就是一个很典型的**简单工厂**。

它负责根据参数帮你构造合适的策略对象。

---

### **5.TorchRolloutEngine**
这是本地PyTorch版本的实现
##### 5.1 初始化

保存：

- `policy_model`
- `tokenizer`
- `device`
- `autocast_ctx`

##### 5.2 `rollout(...)`

```
class TorchRolloutEngine(RolloutEngine):

    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):

        self.policy_model = policy_model

        self.tokenizer = tokenizer

        self.device = device

        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:

        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model

		#不记录梯度，但它不等于inference mode， 也不会自动让张量变成inference tensor
        with torch.no_grad():

            output_ids = model.generate(

                input_ids=prompt_ids,

                attention_mask=attention_mask,

                max_new_tokens=max_new_tokens,

                do_sample=True,

                temperature=temperature,

                num_return_sequences=num_generations,

                pad_token_id=self.tokenizer.pad_token_id,

                eos_token_id=self.tokenizer.eos_token_id,

            )  # [B*num_gen, P+R]

        prompt_len = prompt_ids.size(1)

        completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]

        from contextlib import nullcontext

        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()

        with ctx:

            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1))

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)

	#因为本地 Torch 方案里，引擎自己就持有模型对象；训练后只要换引用即可。说明Torch 版引擎和训练器在同一进程内
    def update_policy(self, model: torch.nn.Module):

        self.policy_model = model
```

**inference_mode**
和 no_grad类似，都是推理时关闭autograd；但它比 `no_grad` 更进一步，还会**禁用 view tracking 和 version counter bump**，因此通常更快；代价是它也更严格：**在这个模式里创建出来的 tensor 不能再拿去参与会被 autograd 记录的计算**。而且它是 **thread-local** 的，只影响当前线程

核心流程是：

1. 取出真正的模型本体  
    如果是 DDP 包装过的，就用 `.module`
2. 在 `torch.no_grad()` 下调用 `model.generate(...)`
3. 从生成结果里切出 completion 部分
4. 再调用 `compute_per_token_logps(...)` 计算生成 token 的 logprob
5. 用 tokenizer 解码生成文本
6. 打包成 `RolloutResult` 返回。

##### 5.3 `update_policy(...)`

直接把内部持有的 `policy_model` 换掉。

---

### **6.`SGLangRolloutEngine` 与工厂函数**
这是远端推理版本
#### 6.1 初始化

```
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
    
		#去掉 URL 末尾多余的 `/`，避免后面拼接口路径时出现 `//generate` 这种情况。
        self.base_url = base_url.rstrip('/')

		#本地共享权重目录。训练进程会把最新权重导出到这里，再通知 SGLang 从这里读
        self.shared_ckpt_path = shared_ckpt_path

        self.timeout = timeout

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.http = requests
```

保存：

- `base_url`
- `shared_ckpt_path`
- `timeout`
- tokenizer
- HTTP 客户端 `requests`

#### 6.2 `rollout(...)`

```
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:

        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []

        for ids, mask in zip(prompt_ids, attention_mask):

            valid_ids = ids[mask.bool()].tolist()

            input_ids_list.append(valid_ids)
            
		#如果每条 prompt 要采样多条 completion，那么传给后端时需要展开成多份输入
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]


		#构造给 SGLang 的请求体
        payload = {

            "input_ids": all_input_ids,
			#控制生成行为
            "sampling_params": {

                "temperature": temperature,

                "max_new_tokens": max_new_tokens,

                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],

            },
			#要求服务端把每个输出 token 的 logprob 一起返回
            "return_logprob": True,

        }

		#发送 POST 请求到 `/generate`
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        
		#如果 HTTP 状态码不是 2xx，直接抛异常
        resp.raise_for_status()

		#把返回体解析成 JSON
        results = resp.json()

		#如果服务端返回单个对象而不是列表，这里统一包成列表，方便后面统一处理
        if not isinstance(results, list):

            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []

        completions = []

        prompt_len = prompt_ids.size(1)

		#用了 `dict.get` 并带默认值，是为了兼容不同返回格式：
		#- 优先从 `meta_info` 里拿
		#- 没有的话再从外层拿
        for i, result in enumerate(results):

            meta = result.get("meta_info", {})

            completion_ids = meta.get("output_ids", result.get("output_ids", []))

            raw_logprobs = meta.get("output_token_logprobs", [])

            logprobs = []

			#清洗 logprob 返回格式
			#说明 SGLang 返回的 `output_token_logprobs` 可能不止一种形态：
			#- 可能是 `[logprob, ...]`
			#- 可能直接就是标量
			#所以这里统一抽出“每个 token 的主 logprob 值”。
            for item in raw_logprobs:

                if isinstance(item, (list, tuple)) and len(item) >= 1:

                    logprobs.append(item[0])

                elif isinstance(item, (int, float)):

                    logprobs.append(item)

            prompt = all_input_ids[i]

            full_output = prompt + completion_ids

            all_output_ids.append(full_output)

            all_completion_ids.append(completion_ids)

            all_logprobs.append(logprobs)

            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device

        max_out_len = max(len(ids) for ids in all_output_ids)

        max_comp_len = max(len(ids) for ids in all_completion_ids)

        max_logp_len = max(len(lp) for lp in all_logprobs)
```

流程是：

1. 根据 `attention_mask` 去掉左侧 padding，只保留有效 prompt token
2. 把每条 prompt 复制 `num_generations` 次
3. 组织 HTTP payload
4. 调 `/generate`
5. 解析返回的 output ids 和 output token logprobs
6. 构造：
    - `full_output = prompt + completion`
    - `completion_ids`
    - `per_token_logps`
7. 因为不同样本长度可能不同，所以统一 padding 成 tensor
8. 返回 `RolloutResult`。

#### 6.3 `update_policy(...)`

```
    def update_policy(self, model: torch.nn.Module):

        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

        abs_path = os.path.abspath(self.shared_ckpt_path)

        unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())

        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}

        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)

        unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight

        self.tokenizer.save_pretrained(abs_path)

        resp = self.http.post(

            f"{self.base_url}/update_weights_from_disk",

            json={"model_path": abs_path},

            timeout=self.timeout

        )

        if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")

        return resp.status_code == 200
```

这是远端方案的关键：

1. 去掉 DDP 包装
2. 取绝对路径
3. 克隆 `lm_head.weight`
4. 把 state_dict 转成 half + cpu
5. `save_pretrained(...)` 导出到共享目录
6. 保存 tokenizer
7. 调 `/update_weights_from_disk`
8. 让远端 SGLang 热更新权重。

#### 6.4 `flush_cache()` / `health()`

分别用于：

- 清 KV cache 或服务缓存
- 健康检查

这说明这个类不只是“能生成”，还负责和服务做运维式交互。

#### 6.5 `create_rollout_engine(...)`

最后的工厂函数根据 `engine_type` 选择创建：

- `TorchRolloutEngine`
- `SGLangRolloutEngine`

这样，上层训练脚本只要配个参数，就能切换后端


整体流程：
**训练器传入 prompt_ids / mask**  
→ **rollout engine 根据配置选择本地或远端推理**  
→ **生成 completion**  
→ **拿到 per-token logprob**  
→ **统一封装成 RolloutResult**  
→ **训练器再拿它去算 reward / advantage / PPO 或 GRPO loss**  
→ **模型更新后调用 update_policy 同步新策略**

---
