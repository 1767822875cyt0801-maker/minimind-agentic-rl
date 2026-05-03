import os
import sys
import random
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dataset.lm_dataset import PretrainDataset, SFTDataset

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

PRETRAIN_PATH = os.path.join(DATASET_DIR, "pretrain_t2t_mini.jsonl")
SFT_PATH = os.path.join(DATASET_DIR, "sft_t2t_mini.jsonl")


def inspect_pretrain(tokenizer):
    print("\n===== PRETRAIN DATASET TEST =====")
    ds = PretrainDataset(tokenizer, PRETRAIN_PATH, max_seq_length=128)
    print("len(pretrain) =", len(ds))

    x, y = ds[1]
    print("type(x) =", type(x), "shape =", tuple(x.shape))
    print("type(y) =", type(y), "shape =", tuple(y.shape))

    print("input_ids[:30] =", x[:30].tolist())
    print("labels[:30]    =", y[:30].tolist())

    pad_id = tokenizer.pad_token_id
    pad_cnt = int((x == pad_id).sum().item())
    supervised_cnt = int((y != -100).sum().item())

    print("pad count =", pad_cnt)
    print("supervised token count =", supervised_cnt)

    print("\nDecoded pretrain sample:")
    print(tokenizer.decode(x.tolist(), skip_special_tokens=False))

    # 基本断言
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 1
    assert y.ndim == 1
    assert supervised_cnt > 0

    print("\n[PASS] PretrainDataset basic checks passed.")


def inspect_sft(tokenizer):
    print("\n===== SFT DATASET TEST =====")
    # 固定随机种子，减少 pre_processing_chat / post_processing_chat 带来的随机性
    random.seed(0)

    ds = SFTDataset(SFT_PATH, tokenizer, max_seq_length=256)
    print("len(sft) =", len(ds))

    x, y = ds[0]
    print("type(x) =", type(x), "shape =", tuple(x.shape))
    print("type(y) =", type(y), "shape =", tuple(y.shape))

    print("input_ids[:50] =", x[:50].tolist())
    print("labels[:50]    =", y[:50].tolist())

    supervised_pos = (y != -100).nonzero(as_tuple=True)[0]
    print("supervised token count =", len(supervised_pos))
    if len(supervised_pos) > 0:
        print("first supervised positions =", supervised_pos[:20].tolist())

    print("\nDecoded SFT sample:")
    print(tokenizer.decode(x.tolist(), skip_special_tokens=False))

    # 只把真正参与 loss 的 token 解码出来看看
    supervised_tokens = [x[i].item() for i in supervised_pos[:120]]
    print("\nFirst supervised text:")
    print(tokenizer.decode(supervised_tokens, skip_special_tokens=False))

    # 基本断言
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 1
    assert y.ndim == 1
    assert len(supervised_pos) > 0
    assert (y == -100).sum().item() > 0  # SFT 应该存在大量不参与 loss 的位置

    print("\n[PASS] SFTDataset basic checks passed.")


def main():
    assert os.path.isdir(MODEL_DIR), f"models dir not found: {MODEL_DIR}"
    assert os.path.isfile(PRETRAIN_PATH), f"pretrain file not found: {PRETRAIN_PATH}"
    assert os.path.isfile(SFT_PATH), f"sft file not found: {SFT_PATH}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    inspect_pretrain(tokenizer)
    inspect_sft(tokenizer)

    print("\nAll lm_dataset tests passed.")


if __name__ == "__main__":
    main()