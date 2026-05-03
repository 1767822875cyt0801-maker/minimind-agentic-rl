import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast




class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)#模型内部预留/支持的位置编码上限
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None#当你想把 RoPE 从原始上下文长度扩到更长时，用来“改位置频率”的缩放规则
        # MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)



#RMSNorm
#LayerNorm：减均值，再除标准差     RMSNorm：不减均值，只按 RMS 缩放
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def norm(self, x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return (self.weight*self.norm(x.float())).type_as(x)
    



#RoPE前后段
#每个i对应一个二维旋转子空间，i = 0,1,2,...,dim/2-1。随着i变大，f(i)变小， 频率变小，周期变长。
#对于一个旋转子空间i，对应一个角频率f(i)。在位置m时，旋转角为θ(m, i) = m * f(i) 。i不变，位置m变大，旋转角变大。令m * f(i)=2π，则m = 2π/f(i)=2π*base**(2i/dim)。这就是该维度i（子空间）对应的周期
#当i变大时，周期变长，则绕一个2π所需的位置长度变得更长。那么对于一个确定长度的序列，越往后的维度越不可能超过周期。虽然高频维度相位增长更快，但在长上下文扩展中，高频更多承担局部位置区分，因此往往尽量少动；低频维度承担更长距离的位置结构，因此更适合被拉伸，去扩展模型的全局上下文范围。
#YaRN先设定一个目标周期 T(i) = orig_max/b。找到那个二维子空间i，它的周期大概等于orig_max / b。使用方法就是：
# YaRN 通过两个阈值 beta_fast 和 beta_slow，
# 分别设定两个目标周期 orig_max / beta_fast 和 orig_max / beta_slow，
# 再利用 inv_dim(b) 反推出对应的二维子空间索引 low 和 high，
# 作为频率缩放的过渡区间边界。小于 low 的部分不变，[low, high] 过渡，大于 high 的完全缩放。
#QA:和直觉相反的，这里缩放的是大于high的频率维度，这一段实际上f(i)小，属于低频段。这是因为YaRN 要解决的是当任务模型从原训练长度扩到更长长度(end)时，哪些频段该尽量保真，哪些频段该承担‘更长距离位置建模’的任务？
#YaRN给出的解决方法是高频更多承担局部、细粒度位置区分；低频更多承担全局、长距离位置结构。如果把高频也大幅压慢，那么会让很多本来用于区分相近 token 位置的快速变化特征，变得没那么敏感，降低模型已经学到的局部位置分辨功能。
#扩张上下文，本质是在说原来模型能稳定表示到orig_max,现在希望它能扩大稳定表示到factor*orig_max，即某些维度的有效周期变长
def precompute_freqs_cis(dim:int, end:int, base:float= 1e6, rope_scaling: dict = None):
    # if rope_scaling is not None:
    #     rope_scaling = {}
    freqs, attn_factor = 1.0/base**(torch.arange(0,dim, 2)[:dim//2].float()/dim), 1.0
    # 初始化默认值，防止 rope_scaling 为 None 时后续引用报错
    orig_max, factor, beta_fast, beta_slow = 2048, 16, 32.0, 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (rope_scaling.get("original_max_position_embeddings", 2048),rope_scaling.get("factor", 16),rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0))
    
    if rope_scaling is not None and end>orig_max:
        #
        inv_dim = lambda b: (dim* math.log(orig_max/(2*b*math.pi)))/(2*math.log(base))
        low, high = max(math.floor(inv_dim(beta_slow)), 0), min(math.ceil(inv_dim(beta_fast)), end-1)
        ramp = torch.clamp((torch.arange(dim//2, device = freqs.device).float() -low)/max(high-low, 0), 0, 1)
        freqs = freqs*(1-ramp+ramp/factor)
    t = torch.arange(end, device=freqs.device)
    cis =  torch.outer(t, freqs)
    cos = torch.cos(cis)
    sin = torch.sin(cis)
    cos = torch.cat([cos, cos], dim= -1)
    sin = torch.cat([sin, -sin], dim= -1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

#cos = torch.cos(θ)  
#cos.shape [end, dim//2]
def apply_rotary_pos_emb(q, k,cos,sin):
    q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
    return q, k


# #RoPE奇偶段
# def precompute_freqs_cis(dim:int, end:int, base:float = 1e6):
#     freqs, attn_factor = 1.0/base**(torch.range(0,dim, 2)[:dim//2].float()/dim), 1.0
#     t = torch.arange(end, device=freqs.device)
#     return torch.outer(t, freqs), attn_factor

# def rotate_half(x):
#     x1 = x[:, ::2]
#     x2 = x[:, 1::2]
#     return torch.cat((-x2, x1), dim = -1)


# def apply_rotary_pos_emb(q, k, cos,sin):
#     cos = torch.repeat_interleave(cos,2,dim=-1)
#     sin = torch.repeat_interleave(sin,2,dim=-1)
#     cos = cos.unsqueeze(0).unsqueeze(0)
#     sin = sin.unsqueeze(0).unsqueeze(0)
#     q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
#     return q, k



def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    """ Repeat kv n_rep times along dim 0"""
    bs, seqlen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:,:,:,None,:].expand(bs, seqlen, num_key_value_heads, n_rep, head_dim).reshape(bs, seqlen, num_key_value_heads*n_rep, head_dim)






class Attention(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.num_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.head_dim = config.head_dim
        self.n_rep = self.num_attention_heads//self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias = False)

        self.q_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        #PyTorch当前版本里有 F.scaled_dot_product_attention且配置里允许使用它，即当前环境和配置允许尝试使用 PyTorch 的 SDPA 快路径
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn


    def forward(self, x, position_embeddings, past_key_value = None, use_cache = False, attn_mask = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq =  xq.view(bsz, seqlen, self.num_attention_heads, self.head_dim)
        xk, xv = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim), xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        #xk.shape = [bsz, seqlen, num_kv_heads, head_dim]->[bsz, total_seqlen, num_kv_heads, head_dim]
        if past_key_value is not None:
            xk, xv = torch.cat([past_key_value[0], xk], dim = 1), torch.cat([past_key_value[1], xv], dim= 1)
        #缓存kv cache，历史 token 的 K/V 直接复用
        past_kv = (xk, xv) if use_cache else None
        #xq.shape = [bsz, num_attention_heads, seqlen, head_dim]
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # print("xq.shape =", xq.shape)
        # print("xk.shape =", xk.shape)
        # print("cos.shape =", cos.shape)
        # print("sin.shape =", sin.shape)
        #把“整段、无 cache、无复杂 mask”的标准 causal attention 交给 PyTorch 的高性能 SDPA；而把“带 cache、带 padding、单步 decode”这些更复杂的场景留给手写分支自己精确控制。
        #self.flash 控制是否允许用这条快路径，self.training 控制 dropout 是否生效，past_key_value 则决定当前是不是在使用 KV cache
        if self.flash and (seqlen >1) and (past_key_value is None) and (attn_mask is None or torch.all(attn_mask==1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask, dropout_p = self.dropout if self.training else 0, is_causal = True)
        else:
            #scores.shape = [bsz, num_attention_heads, seqlen, total_seqlen]
            #两种mask：casual mask 和 attn_mask
            #casual mask 用于控制当前 token 的位置不能参考未来的 token
            #attn_mask 一般是padding mask或者attention mask，不允许模型去关注那些补出来的无效位置
            scores = torch.matmul(xq, xk.transpose(-1, -2)) /math.sqrt(self.head_dim)
            scores[:, :, :, -seqlen:] += torch.full((seqlen, seqlen), float('-inf'), device=scores.device).triu(1)
            #如果 attn_mask存在，attn_mask.shape = [bsz,seqlen],有效位置为1，无效位置为0
            #[bsz,seqlen]->[bsz,1,seqlen]->[bsz,1,1,seqlen]
            #使无效位置的分数变成iNf，softmax后变成0，从而实现mask
            if attn_mask is not None:
                scores += (1.0 - attn_mask.unsqueeze(1).unsqueeze(2))* -1e9
            #attn_dropout在softmax(QK^T/sqrt(d_k))之后，乘以V之前，控制的是某个token对另一个token的关注连接要不要被临时丢掉
            output = self.attn_dropout( F.softmax(scores, dim = -1)).type_as(xq)@xv
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        #rsid_dropout在输出V之后，当前 attention 子层输出的哪些特征分量要被临时丢掉
        output = self.resid_dropout(self.o_proj(output))
        #past_kv.shape = [bsz, total_seqlen, num_kv_heads, head_dim]
        return output, past_kv



#普通 MLP 前馈层(FFN),此时x.shape = [bsz, seqlen, hidden_size]
# Attention层的本质是在计算token之间的权重，对value进行加权和
#FFN层对每个 token 单独做非线性映射，把隐藏空间扩展到更高维，再压回去，学到更复杂的特征组合
#SwiGLU
#FFN(x)=W_down​(ϕ(W_gate(x))⊙(W_up(​x)))
class FeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig,intermediate_size:int= None):
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias = False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias = False)
        #从 Hugging Face 的 ACT2FN 里取激活函数
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        return  self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))



#MOE     use_moe = True   加了专家的FFN
# 每个 token
#     ↓
# gate 计算属于哪些 expert
#     ↓
# 送到 top-k 个 FeedForward expert
#     ↓
# 加权合并输出
class MOEFeedForwad(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias= False)
        self.experts - nn.ModuleList([FeedForward(config,intermediate_size= config.intermediate_size) for _ in range(config.num_experts)])
    
    # x.shape = [bsz, seqlen, hidden_size]
    def forward(self, x):
        bsz, seqlen, hidden_dim = x.shape
        #针对每个token计算，把前两维展平
        x_flat = x.view(-1, hidden_dim)
        #gate.shape = [bsz*seqlen, num_experts]
        scores = F.softmax(self.gate(x_flat), dim = -1)
        #topkweight.shape = [bsz*seqlen, k] 每个token的前k个最高分的专家
        topk_weight, topk_idx = torch.ropk(scores, k = self.config.num_experts_per_tok, dim = -1)
        if self.config.norm_topk_prob:
            topkweight = topkweight / (topkweight.sum(dim = -1, keepdim = True) + 1e-20)
        
        #设置输出
        y = torch.zeros_like(x_flat)

        for i,expert in enumerate(self.experts):
            #mask.shape = [bsz*seqlen, k],里面的值全是0或者1
            #mask[n,j]代表第 n 个 token 的第 j 个 top-k expert 正好是当前 expert i
            #相当于原本topk_idx里的值，如果等于i，则值为1，否则为0
            mask = topk_idx == i
            #如果存在一个True，则需要处理
            if mask.any():
                #mask.any(dim = -1) 从mask中选出所有含有True的行
                #.nonzero()这些选出行的索引
                #.flatten()把多维的索引变成一维的索引
                #token_idx.shape = [num_tokens_with_i]
                token_idx = mask.any(dim = -1).nonzero().flatten()
                #tok_weight.shape = [bsz*seqlen, k]
                #用布尔 mask 索引后，会取出所有为 True 的那些权重
                #weight.shape = [num_tokens_with_i, 1]
                weight = topk_weight[mask].view(-1,1)
                #x_flat[token_idx] 从所有 token 中选出当前 expert 负责的那些 token
                #y.index_add_(0, token_idx, ...) 按照 token 的原始索引，把当前 expert 的贡献加回到总输出 y 的对应行
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]))*weight).to(y.dtype)
            #没有一恶搞token选择这个专家
            elif self.training:
                #y[0,0] 只是随便找一个标量位置挂一下，不影响数值
                y[0, 0] = 0 * sum(p.sum() for p in self.parameters())
        #只在训练阶段并且配置里启用了辅助损失时计算
        #平衡expert被选中的概率
        if self.training and self.config.router_aux_loss_coef>0:
            #one-hot 后 topk_idx.shape 变成[B*S, k, E]
            #.mean(0)在token维度上求平均，[k, E]   在第i个topk位置上，各个专家被选中的概率
            load = F.one_hot(topk_idx, self.config.num_experts).folat().mean(0)
            #同时考虑“硬选择频率”和“软路由概率”的一致性/均衡性
            self.aux_loss = (load * scores.mean(0)).sum()* self.config.num_experts * self.config.router_aux_loss_coef
        else:
            #scores.new_zeros(...) 的好处是device 一致 dtype 一致
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(bsz, seqlen, hidden_dim), self.aux_loss


        




    


#典型的Pre-Norm Transformer block
#hidden_states
#   ↓
#RMSNorm
#   ↓
#Attention
#   ↓
#残差相加(加上最开始进入这一层的hidden_states)
#   ↓
#RMSNorm
#   ↓
#FeedForward / MoEFeedForward
#   ↓
#残差相加(加入上一次残差相加的结果)

#每次残差相加的都是上一次RMSNorm的之前的输入

#[bsz, seqlen, hidden_size] -> [bsz, seqlen, hidden_size]
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int,config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.input_attention_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = MOEFeedForwad(config) if config.use_moe else FeedForward(config, intermediate_size= config.intermediate_size)

    #hidden_states.shape = [bsz, seqlen, hidden_size]
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(self.input_attention_norm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)
        hidden_states += residual
        residual = hidden_states
        mlp_out = self.mlp(self.post_attention_norm(hidden_states))
        if isinstance(mlp_out, tuple):   # 兼容 MoE
            mlp_out, _ = mlp_out
        hidden_states = residual + mlp_out 
        return hidden_states, present_key_value



#[bsz, seqlen] -> [bsz, seqlen, hidden_size]
class MiniMindModel(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.config = config 
        self.vocab_size, self.hidden_size = config.vocab_size, config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l,config) for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        fres_cos, fres_sin = precompute_freqs_cis(dim=config.head_dim, end = config.max_position_embeddings, base = config.rope_theta, rope_scaling= config.rope_scaling)
        self.register_buffer("fres_cos", fres_cos, persistent=False)
        self.register_buffer("fres_sin", fres_sin, persistent=False)
    

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        bsz, seqlen = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        if past_key_values[0] is not None and past_key_values[0][0] is not None:
            start_pos = past_key_values[0][0].shape[1]
        else:
            start_pos = 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (
    self.fres_cos[:, :, start_pos:start_pos + seqlen, :].transpose(1, 2),
    self.fres_sin[:, :, start_pos:start_pos + seqlen, :].transpose(1, 2),
)

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForwad)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


#GenerationMixin 继承了 PreTrainedModel,给具体的语言模型提供.generate()能力的一层通用生成框架
#模型自己负责forward计算logits，GenerateMixin负责把“输入准备 → cache 管理 → 解码策略 → 停止条件 → 输出组织”整套流程串起来
class MiniMindForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = MiniMindConfig
    def __init__(self, config:MiniMindConfig):
        super().__init__(config)
        self.config = config
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False)
        self.model.embed_tokens.weight = self.lm_head.weight
    

    #in_puts.shape = [bsz, seqlen]
    #输出.shape = [bsz, seqlen, vocab_size]
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, presents, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        slice_start = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_start, :])

        loss = None
        if labels is not None:
            x = logits[:, :-1, :].contiguous()
            y = labels[:, 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )
    #纯推理模式，不记录梯度，更适合生成阶段
    @torch.inference_mode()
    def generate(self, inputs = None, attention_mask = None, max_new_token = 8192, temperature = 0.85, top_p = 0.85, top_k = 50, eos_token_id = 2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0,**kwargs):
        #从kwargs中获取input_ids，没有的话就用inputs
        #复制成多条
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        #记录哪些样本已经结束
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        #如果支持流式输出，先把prompt输出到streamer
        if streamer:
            streamer.put(input_ids.cpu())
        
        #主循环,每轮只生成一个token
        for _ in range(max_new_token):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            #第一次循环：Prefill，past_key_values为None，past_len为0。如果inputs_ids.shape[1] = [B, L0],那么第一次forward的input_ids.shape[1] = [B, L0]
            #第二次循环，缓存长度变成L0，而前一轮已经把一个新的token拼到了input_ids末尾，所以input_ids.shape[1] = [B, L0+1]。那么input_ids[:, past_len:] = input_ids[:, L0:]，只剩最后那个新的token，shape变成[B, 1]，进入了decode
            #第三轮以及之后同理input_ids[:, past_len:] -> 只剩最新 token -> decode
            outputs = self.forward(
                input_ids[:, past_len:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            #把mask扩展一个维度 attention_mask.shape = [B, L+1]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1) if attention_mask is not None else None
            #无论 prefill 还是 decode，真正用于决定“下一个 token 是什么”的，永远都是当前输出序列最后一个位置的 logits
            #outputs.shape = [bsz, seqlen, vocab_size]
            #温度缩放
            logits = outputs.logits[:, -1, :]/temperature

            #对已经在当前序列中出现过的 token，把它们的 logits 降低
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            #logits.shape = [B,V]
            #torch.topk(logits, k=top_k)[0] 的shape = [B,top_k]
            #找每行最大的前 top_k 个值,取这 top_k 中最小的那个，作为阈值,小于阈值的 logits 全设成 -inf
            if top_k > 0:
                logits[logits < torch.topk(logits, k=top_k)[0][:, -1, None]] = -float("inf")

            #nucleus sampling
            #先把所有 token 按概率从大到小排序
            # 从概率最大的 token 开始累加概率
            # 只保留“累计概率刚好覆盖到 top_p”这一小撮 token
            # 其余 token 全部屏蔽掉（logits 设为 -inf）
            #最后只在保留下来的这部分 token 里采样
            if top_p < 1.0:
                #按 logits 从大到小排序,sorted_logits: 排序后的值,sorted_indices: 原 vocab 下标.shape = [B,V]
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                #先 softmax，再做累积概率。累积概率没超过 top_p 的位置：False
                mask = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)>top_p
                #右移一位，保留第一个超过阈值的 token
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                #把排序空间里的 mask，映射回原 vocab 下标空间，然后把这些位置的 logits 设成 -inf
                #scatter按照给定下标，把值散射回目标张量对应位置
                #输入：排序空间里的mask，索引：sorted_indices，输出：原vocab顺序下的logits
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            
            #采样或贪心选 token，Flase 为贪心，True为随机采样
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            #eos_token_id 里的 eos 是End Of Sequence,序列结束标记 token 的 id
            #对于那些已经结束的样本，不再使用本轮真正采样出来的 token，而是强制把它们的 next_token 改成 eos_token_id。 shape 还能保持整齐,已结束样本不会继续胡乱生成别的词
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                #把这一轮刚刚生成出 EOS 的样本，标记为 finished=True
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

            





        



















