import json
import re
from pathlib import Path

src = Path("dataset/sft_t2t_mini.jsonl")
dst = Path("dataset/sft_t2t_clean_v1.jsonl")

meta_phrases = [
    "回答上面的问题",
    "基于以上这段文本内容回答",
    "根据以上内容回答",
    "我需要保持友好和温暖",
    "首先，我需要确认",
    "用户可能想",
    "接下来，我要考虑",
    "我需要更多上下文信息才能回答这个问题",
    "我作为AI模型，无法",
    "好的，现在",
    "请在文章中引入",
]

bad_identity_phrases = [
    "我是MiniMind",
    "我是由jingyaogong",
    "由jingyaogong创建",
    "由jingyaogong开发",
    "高效小参数AI模型",
]

def clean_assistant_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 去掉显式 think 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 去掉元提示污染
    for p in meta_phrases:
        text = text.replace(p, "")

    # 清理多余空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def should_drop_sample(convs):
    # 至少要有 assistant
    has_assistant = any(t.get("role") == "assistant" for t in convs)
    if not has_assistant:
        return True

    for t in convs:
        if t.get("role") == "assistant":
            txt = (t.get("content") or "").strip()

            # 空/近空回答
            if txt in {"", "。", "，", "？", "!", "！"}:
                return True

            # 编码坏掉
            if "�" in txt:
                return True

            # 很短且只有身份模板
            if any(p in txt for p in bad_identity_phrases) and len(txt) < 40:
                return True

    return False

kept = 0
dropped = 0

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
    for line in fin:
        obj = json.loads(line)
        convs = obj.get("conversations", [])
        if not isinstance(convs, list) or len(convs) == 0:
            dropped += 1
            continue

        new_convs = []
        for turn in convs:
            turn = dict(turn)

            # 关键：清空 reasoning_content
            if "reasoning_content" in turn:
                turn["reasoning_content"] = ""

            # 清洗 assistant 内容
            if turn.get("role") == "assistant":
                turn["content"] = clean_assistant_text(turn.get("content", ""))

            new_convs.append(turn)

        if should_drop_sample(new_convs):
            dropped += 1
            continue

        obj["conversations"] = new_convs
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("kept =", kept)
print("dropped =", dropped)
print("saved to", dst)