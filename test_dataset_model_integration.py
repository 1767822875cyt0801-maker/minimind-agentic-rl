import os
import sys
import random
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset, SFTDataset

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

PRETRAIN_PATH = os.path.join(DATASET_DIR, "pretrain_t2t_mini.jsonl")
SFT_PATH = os.path.join(DATASET_DIR, "sft_t2t_mini.jsonl")


def build_model(device):
    config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        flash_attn=False,   # CPU 测试时关掉更稳
    )
    model = MiniMindForCausalLM(config).to(device)
    model.eval()
    return model


def test_pretrain_forward(model, tokenizer, device):
    print("\n===== PRETRAIN -> MODEL TEST =====")
    ds = PretrainDataset(tokenizer, PRETRAIN_PATH, max_seq_length=128)
    x, y = ds[0]

    x = x.unsqueeze(0).to(device)   # [1, 128]
    y = y.unsqueeze(0).to(device)   # [1, 128]

    with torch.no_grad():
        outputs = model(input_ids=x, labels=y)

    print("input_ids.shape =", tuple(x.shape))
    print("labels.shape    =", tuple(y.shape))
    print("loss =", outputs.loss.detach().item())
    print("logits.shape =", tuple(outputs.logits.shape))

    assert outputs.loss is not None
    assert outputs.logits.shape == (1, 128, tokenizer.vocab_size)
    print("[PASS] Pretrain dataset -> model forward passed.")


def test_sft_forward(model, tokenizer, device):
    print("\n===== SFT -> MODEL TEST =====")
    random.seed(0)
    ds = SFTDataset(SFT_PATH, tokenizer, max_seq_length=256)
    x, y = ds[0]

    x = x.unsqueeze(0).to(device)   # [1, 256]
    y = y.unsqueeze(0).to(device)   # [1, 256]

    with torch.no_grad():
        outputs = model(input_ids=x, labels=y)

    supervised_cnt = int((y != -100).sum().item())

    print("input_ids.shape =", tuple(x.shape))
    print("labels.shape    =", tuple(y.shape))
    print("supervised token count =", supervised_cnt)
    print("loss =", outputs.loss.detach().item())
    print("logits.shape =", tuple(outputs.logits.shape))

    assert outputs.loss is not None
    assert supervised_cnt > 0
    assert outputs.logits.shape == (1, 256, tokenizer.vocab_size)
    print("[PASS] SFT dataset -> model forward passed.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = build_model(device)

    test_pretrain_forward(model, tokenizer, device)
    test_sft_forward(model, tokenizer, device)

    print("\nAll dataset-model integration tests passed.")


if __name__ == "__main__":
    main()