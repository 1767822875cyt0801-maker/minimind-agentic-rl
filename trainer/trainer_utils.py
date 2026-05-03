"""
训练工具函数集合
"""
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import math
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from models.model_minimind import MiniMindForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models') 
OUT_DIR = os.path.join(PROJECT_ROOT, 'out')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

tokenizer_path = MODEL_DIR
#判断是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


#只在主进程打印日志
def Logger(content):
    if is_main_process():
        print(content)

#统计模型参数量，尤其兼容 MoE 场景，区分总参数量和 active 参数量
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'num_experts', 0)
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_experts', 0)

    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    base = total - (expert*n_routed + shared_expert*n_shared)
    active = base + expert*n_active+shared_expert*n_shared

    if active < total:
        Logger(f'Model Params:{total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params:{total:.2f}M')

#按当前 step 计算学习率。这里实现的是一个 cosine 风格衰减公式
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

#初始化分布式训练环境.它通过环境变量判断是不是 DDP，如果是就初始化进程组并设置当前 GPU
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


#统一设置 Python / NumPy / Torch / CUDA 的随机种子，尽量保证实验可复现
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#保存模式：传入 model / optimizer / scaler / epoch / step 时，把：模型权重优化器状态、scaler 状态、epoch / step、wandb id、world size、一起保存下来
#加载模式：不传 model 时，它去检查 resume 文件是否存在，若存在就读出来，必要时还会根据 world size 的变化修正 step
def lm_checkpoint(lm_config, weight = 'full_sft', model = None, optimizer = None, epoch = 0, step = 0, wandb = None, save_dir = CHECKPOINT_DIR,**kargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    
    #保存模型
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        state_dict = raw_model.state_dict()
        state_dict = {k:v.half().cpu() for k,v in state_dict.items()}

        ckp_path_tmp = ckp_path+'.tmp'
        #保存纯权重到ckp_path_tmp，写完后原子性替换到ckp_path，避免保存过程中被读取到不完整的文件
        torch.save(state_dict, ckp_path_tmp)
        os.replace(ckp_path_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        resume_data = {
            'model':state_dict,
            'optimizer':optimizer.state_dict(),
            'epoch':epoch,
            'step':step,
            'wandb_id':wandb_id,
            'world_size':dist.get_world_size() if dist.is_initialized() else 1

        }
        for k,v in kargs.items():
            if v is not None:
                if hasattr(v, 'state_dict'):
                    raw_model = v.module if isinstance(v, DistributedDataParallel) else v
                    resume_data[k] = raw_model.state_dict()
                else:
                    resume_data[k] = v
        
        resume_path_tmp = resume_path+'.tmp'

        #保存其他状态到resume_path_tmp，写完后原子性替换到resume_path，避免保存过程中被读取到不完整的文件
        #resume_data 保存了模型、优化器、epoch、step、wandb id、world size，以及其他通过 kargs 传入的状态（比如 scaler）
        torch.save(resume_data, resume_path_tmp)
        os.replace(resume_path_tmp, resume_path)
        del resume_data, state_dict
        # torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #加载模型
    else:
        if os.path.exists(resume_path): 
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            cur_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != cur_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // cur_ws
                Logger(f'Warning: world size changed from {saved_ws} to {cur_ws}, step is reduced to {ckp_data["step"]}')
            ckp_data['world_size'] = cur_ws
            return ckp_data
        return None

#构造模型，加载tokenizer，必要时加载预训练权重，并统计参数量。返回模型和tokenizer
#输出模型和tokenizer
def init_model(lm_config, from_weight = 'pretrain', tokenizer_path = MODEL_DIR, save_dir = OUT_DIR, device = 'cuda'):
    model = MiniMindForCausalLM(lm_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if from_weight is not None:
        moe = '_moe' if lm_config.use_moe else ''
        weight_path = os.path.join(save_dir, f'{from_weight}_{lm_config.hidden_size}{moe}.pth')
        assert os.path.isfile(weight_path), f'Weight not found: {weight_path}'
        weights = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weights)
        # Logger(f'Load weight from {weight_path}')
    
    get_model_params(model, lm_config)
    Logger(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M')
    return model.to(device), tokenizer

#在断点续训时，跳过已经训练的 batchcal
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch_cnt = []
        skipped_batch_num = 0
        for idx in self.sampler:
            batch_cnt.append(idx)
            if len(batch_cnt)==self.batch_size:
                if skipped_batch_num < self.skip_batches:
                    skipped_batch_num += 1
                    batch_cnt = []
                else:
                    yield batch_cnt
                    batch_cnt = []
        if len(batch_cnt)>0 and skipped_batch_num >= self.skip_batches:
                yield batch_cnt
    

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size -1)//self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device = 'cuda', dtype = torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def get_score(self, messages, responces):
        history_text = '\n'.join([f'{m["role"]}: {m["content"]}' for m in messages[:-1]])
        last_quary = messages[-1]['content'] if messages else ''
        messages_context = f"{history_text}\n以上是对话历史。新问题是:\n{last_quary}" if history_text else last_quary
        eval_messages = [
            {"role": "user", "content": messages_context},
            {"role": "assistant", "content": responces},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)

        


