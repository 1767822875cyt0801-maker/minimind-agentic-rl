from pathlib import Path
from itertools import islice

src = Path("dataset/sft_t2t_clean_v1.jsonl")
dst = Path("dataset/sft_t2t_clean_v1_1k.jsonl")

with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8", newline="\n") as g:
    for line in islice(f, 1000):
        g.write(line)