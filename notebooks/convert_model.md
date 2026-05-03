模型格式转换与导出工具脚本
负责把 **MiniMind 自己的 PyTorch 权重**、**Transformers/HuggingFace 生态格式**、**LoRA 合并后的基模权重**、以及 **chat template 的 jinja/json 表示** 互相转换

## 模块功能
**1.环境准备与导入**
```
import os
import sys
import json

__package__ = "scripts"
sys.path.append(...)
import torch
import transformers
import warnings
from transformers import ...
from model.model_minimind import ...
from model.model_lora import ...

warnings.filterwarnings(...)
```

**2.MiniMind 原生结构 → Transformers-MiniMind 格式**

- 用 **MiniMind 自己的模型类** 重新构造模型
- 加载 `.pth`
- 保存成 `save_pretrained` 风格目录
- 连同 tokenizer 一起导出
- 兼容 transformers 5.0 的配置差异

```
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")
```

**3.MiniMind 权重 → Qwen/Transformers 兼容格式**
- 把 MiniMind 权重映射到 `Qwen3ForCausalLM` 或 `Qwen3MoeForCausalLM`
- 让模型更容易接入 HuggingFace 通用生态
- 对 MoE 权重做结构适配
- 保存 tokenizer 和 config


**4.Transformers 格式 → 普通 PyTorch `.pth`**
反向转换器
- 从 HuggingFace 风格目录读模型
- 抽出 `state_dict`
- 存成普通 `.pth`

**5.LoRA 合并导出**
- 加载基础 MiniMind 权重
- 注入 LoRA 结构
- 读取 LoRA adapter
- 合并到 base model
- 导出合并后的普通 PyTorch 权重


**6.chat template 的 jinja/json 互转 + 主程序入口**
