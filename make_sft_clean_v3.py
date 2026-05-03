import json
import re
from pathlib import Path

SRC = Path("dataset/sft_t2t_mini.jsonl")
DST = Path("dataset/sft_t2t_clean_v3.jsonl")

THINK_PAT = re.compile(r"<think>|</think>", re.IGNORECASE)

DROP_USER_KEYWORDS = [
    "真实来源", "真实身份", "身份来自哪里", "开发背景", "由谁创建", "由谁开发",
    "jingyaogong", "minimind", "开源模型训练",
    "今天过得怎么样", "最近有没有", "看过什么电影", "看过什么好书",
    "读过什么小说", "遇到什么挑战", "有趣的事", "有没有关注过"
]
DROP_USER_KEYWORDS += [
    "诞生于", "哪款框架", "什么技术",
    "真实开源代码训练", "特定开源代码库训练", "特定数据集训练",
    "背后的模型", "模型是哪个版本", "开发团队", "团队背景",
    "qwen", "阿里云",
    "你最近在忙什么", "最近在优化哪些具体技术",
    "你最喜欢的音乐类型", "你希望学习哪些具体技能",
    "你感兴趣的具体技能或领域", "你今天想不想",
    "你猜下次AI会怎么改变我们的生活"
]

DROP_USER_KEYWORDS += [
    "真实数据训练",
    "开源模型进行训练",
    "背后的算法",
    "哪个公司开发",
    "支持多语言翻译",
    "支持实时翻译",
    "模型是否支持",
    "你今天想做什么",
    "你今天的任务是什么",
    "你今天最想了解什么",
]

DROP_ANYWHERE_KEYWORDS = [
    "MiniMind",
    "minimind",
    "jingyaogong",
    "由jingyaogong创建",
    "由jingyaogong开发",
    "由 Jingyaogong 创造",
    "高效小参数AI模型",
]

DROP_ASSISTANT_KEYWORDS = [
    "我作为AI模型",
    "我没有个人经历",
    "我没有情感",
    "我没有观看电影的能力",
    "我没有阅读书籍的能力",
    "我无法体验",
    "我无法理解和处理",
    "我需要更多上下文信息才能回答这个问题",
]

DROP_ASSISTANT_KEYWORDS += [
    "我基于开源模型训练",
    "我基于真实数据进行训练",
    "我支持多语言翻译",
    "我支持实时翻译",
    "我的任务是提供准确",
    "我今天将继续提供高效",
    "我专注于多领域知识整合",
]

DROP_ASSISTANT_KEYWORDS += [
    "Qwen", "阿里云人工智能实验室", "阿里云开发的Qwen系列",
    "Transformer架构的高效框架",
    "我正在优化模型性能",
    "我正在优化模型的训练效率",
    "我作为AI模型，没有观看电影的能力",
    "我作为AI模型，没有阅读书籍的能力",
    "我无法拥有个人喜好",
]

META_PHRASES = [
    "回答上面的问题",
    "基于以上这段文本内容回答",
    "根据以上内容回答",
    "我需要保持友好和温暖",
    "首先，我需要确认",
    "用户可能想",
    "接下来，我要考虑",
    "好的，现在",
    "请在文章中引入",
]

def contains_think(text: str) -> bool:
    return bool(THINK_PAT.search(text or ""))

def hit_any(text: str, keywords) -> bool:
    text = (text or "").lower()
    return any(k.lower() in text for k in keywords)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 去显式 think 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 去元提示污染
    for p in META_PHRASES:
        text = text.replace(p, "")

    # 清理多余空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def has_repeated_adjacent_users(convs):
    last_user = None
    for turn in convs:
        if turn.get("role") == "user":
            cur = (turn.get("content") or "").strip()
            if last_user is not None and cur == last_user:
                return True
            last_user = cur
    return False

def should_drop_sample(convs):
    if not any(t.get("role") == "assistant" for t in convs):
        return True

    for turn in convs:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        reasoning = (turn.get("reasoning_content") or "").strip()

        # 1) 任何 think 直接丢
        if contains_think(content) or contains_think(reasoning):
            return True

        # 2) 任意位置出现身份模板，直接丢
        if hit_any(content, DROP_ANYWHERE_KEYWORDS) or hit_any(reasoning, DROP_ANYWHERE_KEYWORDS):
            return True

        # 3) user 含身份/人格闲聊问题，直接丢
        if role == "user" and hit_any(content, DROP_USER_KEYWORDS):
            return True

        # 4) assistant 含 AI 人格/拒答模板，直接丢
        if role == "assistant" and hit_any(content, DROP_ASSISTANT_KEYWORDS):
            return True

        # 5) 空/近空回答
        if role == "assistant" and content in {"", "。", "，", "？", "!", "！"}:
            return True

        # 6) 编码损坏
        if "�" in content:
            return True

    # 7) 相邻重复 user
    if has_repeated_adjacent_users(convs):
        return True

    return False

def main():
    kept, dropped = 0, 0

    if not SRC.exists():
        raise FileNotFoundError(f"未找到源文件: {SRC}")

    with SRC.open("r", encoding="utf-8") as fin, DST.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            convs = obj.get("conversations", [])
            if not isinstance(convs, list) or not convs:
                dropped += 1
                continue

            new_convs = []
            for turn in convs:
                turn = dict(turn)

                # 清空 reasoning_content
                if "reasoning_content" in turn:
                    turn["reasoning_content"] = ""

                # 清洗 content
                if "content" in turn:
                    turn["content"] = clean_text(turn["content"])

                new_convs.append(turn)

            if should_drop_sample(new_convs):
                dropped += 1
                continue

            obj["conversations"] = new_convs
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"saved to: {DST}")
    print(f"kept   = {kept}")
    print(f"dropped= {dropped}")

if __name__ == "__main__":
    main()