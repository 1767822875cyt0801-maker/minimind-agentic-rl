Q1：train_pretrain.py中的

lm_config = MiniMindConfig(

    hidden_size=args.hidden_size,

    num_hidden_layers=args.num_hidden_layers,

    use_moe=args.use_moe,

    num_attention_heads=8,

    num_key_value_heads=4,

    flash_attn=True,

     max_position_embeddings=512,

)
注释掉max_position_embeddings=512,会出现loss: nan, logits_loss: nan

A1：
注释掉以后，`MiniMindConfig` 会回到默认值：
self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
而 `MiniMindModel` 在初始化时会立刻用这个值去预计算整张 RoPE 的 `fres_cos / fres_sin` 表，然后注册成 buffer。也就是说，`max_position_embeddings` 不只是“缓存更长的位置表”，它直接决定了 `precompute_freqs_cis()` 走哪条逻辑
```
orig_max, factor, beta_fast, beta_slow = 2048, 16, 32.0, 1.0
...
if end > orig_max:
    ...
    freqs = freqs * (1 - ramp + ramp / factor)
```
默认 `orig_max` 是 2048。于是：

- 当设 `max_position_embeddings=512` 时，`512 <= 2048`，不会进缩放分支；
- 当注释掉后回到默认 `32768` 时，`32768 > 2048`，会进入缩放分支。

但是只要end>orig_max，就会进入长上下文频率缩放分支，而不是“只有显式启用 `rope_scaling` 才进入”。虽然训练时 `max_seq_len` 只有 64，但你真正用到的前 64 个位置的 `cos/sin`，已经不是“普通 RoPE”的前 64 行了，而是“长上下文缩放后 RoPE”的前 64 行。也就是说，**位置编码本身变了**。
在attention前向中，`xq/xk` 会先做：
`xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)`
这一步的`cos/sin` 分布不稳定，后面 attention score、softmax、logits、cross entropy 都可能出现数值异常，最后在第一个 batch 的前向里直接产出 NaN


`max_seq_len` 和 `max_position_embeddings` 不是一回事：
- `max_seq_len` 是“这次喂多长”
- `max_position_embeddings` 是“内部位置表准备多长”
