import json
import re
from pathlib import Path

src = Path("dataset/sft_t2t_mini.jsonl")
dst = Path("dataset/sft_t2t_clean_v2.jsonl")

THINK_PAT = re.compile(r"<think>|</think>", re.IGNORECASE)

DROP_USER_KEYWORDS = [
    "真实来源", "真实身份", "身份来自哪里", "开发背景", "由谁创建", "由谁开发",
    "jingyaogong", "minimind", "开源模型训练",
    "今天过得怎么样", "最近有没有", "看过什么电影", "看过什么好书",
    "读过什么小说", "遇到什么挑战", "有趣的事", "有没有关注过"
]

DROP_ASSISTANT_KEYWORDS = [
    "我是MiniMind",
    "我是由jingyaogong",
    "由jingyaogong创建",
    "由jingyaogong开发",
    "我是由 Jingyaogong",
    "我作为AI模型",
    "我没有个人经历",
    "我没有情感",
    "我没有观看电影的能力",
    "我没有阅读书籍的能力",
]

def contains_think(text: str) -> bool:
    return bool(THINK_PAT.search(text or ""))

def hit_any(text: str, keywords) -> bool:
    text = (text or "").lower()
    return any(k.lower() in text for k in keywords)

def has_repeated_adjacent_users(convs):
    last_user = None
    for turn in convs:
        if turn.get("role") == "user":
            cur = (turn.get("content") or "").strip()
            if last_user is not None and cur == last_user:
                return True
            last_user = cur
    return False

kept, dropped = 0, 0

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
    for line in fin:
        obj = json.loads(line)
        convs = obj.get("conversations", [])
        if not isinstance(convs, list) or not convs:
            dropped += 1
            continue

        drop = False

        for turn in convs:
            role = turn.get("role", "")
            content = turn.get("content", "") or ""
            reasoning = turn.get("reasoning_content", "") or ""

            # 1) 任何 think 直接丢
            if contains_think(content) or contains_think(reasoning):
                drop = True
                break

            # 2) user 里出现身份/人格问题，直接丢
            if role == "user" and hit_any(content, DROP_USER_KEYWORDS):
                drop = True
                break

            # 3) assistant 里出现身份模板/拒答模板，直接丢
            if role == "assistant" and hit_any(content, DROP_ASSISTANT_KEYWORDS):
                drop = True
                break

        # 4) 相邻重复 user 问题，直接丢
        if not drop and has_repeated_adjacent_users(convs):
            drop = True

        if drop:
            dropped += 1
            continue

        # 5) 清空 reasoning_content
        for turn in convs:
            if "reasoning_content" in turn:
                turn["reasoning_content"] = ""

        obj["conversations"] = convs
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("kept =", kept)
print("dropped =", dropped)
print("saved to", dst)