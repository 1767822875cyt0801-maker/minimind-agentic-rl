**给已有模型临时挂上 LoRA 适配器，并负责保存 / 加载 / 合并 LoRA 权重**
完成lora增量

### 和 `model_minimind.py` 的关系
`model_minimind.py` 定义的是**基础模型结构**，比如 attention、MLP、RMSNorm、embedding 等。  
而 `model_lora.py` 并不重新定义这些层，它是**在模型已经构建好以后，给其中一部分 `nn.Linear` 额外挂一个 LoRA 分支**。所以它依附于主体模型，而不是替代主体模型。

### 和训练脚本的关系
比如 `train_full_sft.py`、某个 `train_lora.py`，通常会：
1. 先初始化 MiniMind 基础模型
2. 调 `apply_lora(model)`
3. 冻结原模型大部分参数
4. 只训练 LoRA 的 `A`、`B` 参数
5. 训练后用 `save_lora()` 保存适配器权重。

### 和推理 / 部署脚本的关系
推理时一般有两种方式：
- **方式 1：底座模型 + LoRA 适配器**
    - 先加载基础模型
    - 再 `apply_lora(model)`
    - 再 `load_lora(model, path)`
- **方式 2：合并后单模型部署**
    - 先加载基础模型
    - 调 `merge_lora(model, lora_path, save_path)`
    - 之后直接用合并后的权重推理



### 模块组成
**1.导入**
```
import torch

from torch import optim, nn
```

**2.LoRA 结构定义**
定义一个 LoRA 分支，本质上是两个无 bias 的线性层串起来：
- `A: in_features -> rank`
- `B: rank -> out_features`
```
class LoRA(nn.Module):

    def __init__(self, in_features, out_features, rank):

        super().__init__()

        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小

        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A

        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B

        # 矩阵A高斯初始化

        self.A.weight.data.normal_(mean=0.0, std=0.02)

        # 矩阵B全0初始化

        self.B.weight.data.zero_()

  

    def forward(self, x):

        return self.B(self.A(x))
```

矩阵A高斯初始化原因：LoRA分支一开始不是完全死的，有一个随机方向可供学习
矩阵B全0初始化原因：保证在训练刚开始的时候:self.B(self.A(x)) = 0，使得模型一开始的行为与原行为一致，不会因为加了 LoRA 就立刻扰动输出

**3.给模型注入 LoRA**

遍历模型的所有模块，找到满足条件的 `nn.Linear`，给它挂上 `module.lora`，然后把原始 `forward` 改成：原始输出 + LoRA输出
```
def apply_lora(model, rank=16):

    for name, module in model.named_modules():
		#这份实现只给方阵线性层加 LoRA(因为module.weight.shape[0] == module.weight.shape[1])
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:

            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
			
			#把 LoRA 子模块挂到当前线性层上，命名为module.lora。之后可以通过module.lora访问它，也能被state_dict()正常追踪。即给当前线性层动态增加一个属性lora
            setattr(module, "lora", lora)

            original_forward = module.forward

  

            # 显式绑定

            def forward_with_lora(x, layer1=original_forward, layer2=lora):

                return layer1(x) + layer2(x)

  
			#真正把当前线性层的 `forward` 替换成带 LoRA 的版本。
            module.forward = forward_with_lora
```
显式绑定的原因：
python里非常典型的 **闭包 + 循环变量晚绑定（late binding）** 问题
这个函数体里引用了外层作用域里的变量 `original_forward` 和 `lora`，而这些变量在 `for` 循环里会不断被重新赋值
如果直接写：
```
def forward_with_lora(x):
    return original_forward(x) + lora(x)
```
那么这个函数并不是把“当时那一层的 `original_forward` 和 `lora` 值”拷贝进去了，  
而是 **记住了这两个名字**，等到将来真正调用时，再去外层找它们当前的值。
而等循环结束后，这两个名字通常已经变成了 **最后一次循环** 的那一组对象



**4.LoRA 权重加载 / 保存**
```
def load_lora(model, path):

    state_dict = torch.load(path, map_location=model.device)
	#如果模型在分布式训练里被 `DistributedDataParallel` 包了一层，参数名常常长这样：module.layers.0.attention.wq.lora.A.weight
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

  
	#只处理已经被 `apply_lora` 注入过 LoRA 的层(`load_lora` 默认假设你已经先 `apply_lora(model)`)
    for name, module in model.named_modules():

        if hasattr(module, 'lora'):
			#筛出属于当前模块的 LoRA 参数;把 key 改成适合 `module.lora.load_state_dict()` 的格式,因为`module.lora` 自己内部只认识A.weight
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
			#把当前模块对应的 LoRA 参数加载进 `module.lora`
            module.lora.load_state_dict(lora_state)

  
  

def save_lora(model, path):

    raw_model = getattr(model, '_orig_mod', model)

    state_dict = {}

    for name, module in raw_model.named_modules():

        if hasattr(module, 'lora'):

            clean_name = name[7:] if name.startswith("module.") else name

            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
			#把当前层的 LoRA 参数合并进总字典
            state_dict.update(lora_state)

    torch.save(state_dict, path)
```
save_lora只处理那些已经挂了 LoRA 的层。  
这说明 `load_lora` 的前提通常是：你已经先执行过 `apply_lora(model)`。  
否则模型里没有 `module.lora`，就没法加载

加载名称变化：module.layers.0.attention.wq.lora.A.weight
①去掉`module.`，如果有的话。变成layers.0.attention.wq.lora.A.weight，放入state_dict中
②model.named_modules()遍历模型中的所有子模块，这里的name就是`layers.0.attention.wq`（上面的apply_lora函数已经把 LoRA 子模块挂到当前线性层上，命名为module.lora）
③对于module.lora自己内部只认识A.weight，所以把state_dict中key中的name去掉






**5.LoRA 合并**
把挂在这个 base Linear 上的 LoRA 增量，折叠回这个 base Linear 的权重里。其中base Linear是本来的线性层
```
def merge_lora(model, lora_path, save_path):
	#先把 LoRA 参数加载到模型中
    load_lora(model, lora_path)

    raw_model = getattr(model, '_orig_mod', model)
    
	#先把原始模型参数复制出来，但排除掉 LoRA 参数
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}

    for name, module in raw_model.named_modules():
		#第二个条件 `'.lora.' not in name` 基本上是防御式写法，避免误处理 LoRA 自己内部的线性层 `A` 和 `B`。
        if isinstance(module, nn.Linear) and '.lora.' not in name:

            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
			#如果当前线性层挂过 LoRA，就进行权重合并。
            if hasattr(module, 'lora'):

                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()

    torch.save(state_dict, save_path)
```