import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model, lm_checkpoint, OUT_DIR, CHECKPOINT_DIR


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        flash_attn=False,
    )

    # 1) init_model 不给任何已有权重时，trainer_utils.py 能不能只靠配置把模型和 tokenizer 正常建出来
    model, tokenizer = init_model(config, from_weight=None, device=device)
    print("init_model passed")
    print("tokenizer vocab size =", tokenizer.vocab_size)

    # 2) checkpoint save/load
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #用lm_checkpoint(...)保存一次模型，测试lm_checkpoint(...)保存逻辑有无问题
    lm_checkpoint(
        config,
        weight="debug_test",
        model=model,
        optimizer=optimizer,
        epoch=1,
        step=7,
    )
    print("checkpoint save passed")

    #再调用一次lm_chaeckpoint(...)读回来，测试lm_checkpoint(...)加载逻辑有无问题
    ckp_data = lm_checkpoint(config, weight="debug_test", model=None)
    assert ckp_data is not None
    print("checkpoint load passed")
    print("loaded keys =", ckp_data.keys())
    print("loaded epoch =", ckp_data["epoch"])
    print("loaded step =", ckp_data["step"])

    print("All trainer_utils tests passed.")


if __name__ == "__main__":
    main()