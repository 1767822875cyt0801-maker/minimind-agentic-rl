import os
import sys
sys.path.append(os.path.abspath("."))

from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset,pre_processing_chat,post_processing_chat

tokenizer = AutoTokenizer.from_pretrained("models")
ds = SFTDataset("dataset/sft_t2t_clean_v5_200.jsonl", tokenizer, max_seq_length=256)

for i in range(10):
    sample = ds.samples[i]
    conversations = sample["conversations"]
    conversations = pre_processing_chat(conversations, add_system_ratio=0.0)
    prompt = ds.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)
    decoded = tokenizer.decode(tokenizer(prompt).input_ids[:256], skip_special_tokens=False)
    print("\n[TOKENIZED-DECODED PREVIEW]")
    print(decoded[:1000])
    print("=" * 80)
    print(f"INDEX = {i}")
    print(prompt[:2000])