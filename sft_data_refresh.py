import json
import random
from pathlib import Path

src = Path("dataset/sft_t2t_mini.jsonl")
dst = Path("dataset/sft_t2t_audit_300.jsonl")

random.seed(42)

with src.open("r", encoding="utf-8") as f:
    lines = f.readlines()

sampled = random.sample(lines, 300)

with dst.open("w", encoding="utf-8", newline="\n") as f:
    for line in sampled:
        f.write(line)