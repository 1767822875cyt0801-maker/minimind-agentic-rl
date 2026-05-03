import torch

# 按你项目里的真实导入路径改
from models.model_minimind import MiniMindConfig, MiniMindForCausalLM


def build_tiny_config():
    """
    用一个很小的配置做 smoke test，CPU 也能较快跑通。
    如果你的 MiniMindConfig 字段名和这里略有不同，
    就按你自己 model_minimind.py 里的定义改一下。
    """
    config = MiniMindConfig(
        vocab_size=6400,
        num_hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        use_moe=False,
    )
    return config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # 1) 构造一个很小的配置，先验证模型逻辑本身
    config = build_tiny_config()

    # 2) 实例化模型
    model = MiniMindForCausalLM(config).to(device)
    print("model init ok")
    print("device:", device)

    # 3) 随机造一批 input_ids
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device
    )

    # labels 先直接拷贝 input_ids，测试 loss 计算链路
    labels = input_ids.clone()

    # 4) eval 模式下 forward（不带梯度）
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    print("\n=== forward with labels ===")
    print("loss:", float(outputs.loss))
    print("logits.shape:", tuple(outputs.logits.shape))

    # 一些基本断言
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert outputs.loss.ndim == 0
    print("forward shape check passed")

    # 5) 再测一次不带 labels 的 forward
    with torch.no_grad():
        outputs_no_label = model(input_ids=input_ids)

    print("\n=== forward without labels ===")
    print("logits.shape:", tuple(outputs_no_label.logits.shape))
    assert outputs_no_label.logits.shape == (batch_size, seq_len, config.vocab_size)
    print("forward(no labels) check passed")

    # 6) train 模式下测 backward
    model.train()
    outputs_train = model(input_ids=input_ids, labels=labels)
    loss = outputs_train.loss
    loss.backward()

    total_grad_params = 0
    non_none_grad_params = 0
    for p in model.parameters():
        total_grad_params += 1
        if p.grad is not None:
            non_none_grad_params += 1

    print("\n=== backward ===")
    print("train loss:", loss.detach().item())
    print("params with grad:", non_none_grad_params, "/", total_grad_params)
    assert non_none_grad_params > 0
    print("backward check passed")

    print("\nAll basic tests passed.")


if __name__ == "__main__":
    main()