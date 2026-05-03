from pathlib import Path
from itertools import islice

src = Path("dataset/sft_t2t_clean_v2.jsonl")
dst = Path("dataset/sft_t2t_clean_v2_300.jsonl")

with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8", newline="\n") as g:
    for line in islice(f, 300):
        g.write(line)

print("saved to:", dst)