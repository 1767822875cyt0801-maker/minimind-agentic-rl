import json
import re
from pathlib import Path
from typing import List, Dict, Any

SRC = Path("dataset/sft_t2t_mini.jsonl")
DST = Path("dataset/sft_t2t_clean_v4.jsonl")

THINK_PAT = re.compile(r"<think>|</think>", re.IGNORECASE)

# -----------------------------
# 1) 高风险 user 主题：整条丢弃
# -----------------------------
DROP_USER_KEYWORDS = [
    # 身份 / 来源 / 框架 / 团队 / 训练来源
    "真实来源", "真实身份", "身份来自哪里", "开发背景", "由谁创建", "由谁开发",
    "背后的模型", "模型是哪个版本", "开发团队", "团队背景",
    "诞生于", "哪款框架", "什么技术",
    "真实开源代码训练", "特定开源代码库训练", "特定数据集训练",
    "真实数据训练", "开源模型进行训练", "开源模型训练",
    "qwen", "阿里云", "jingyaogong", "minimind",

    # AI 人格/自我经历/偏好/状态
    "今天过得怎么样", "最近有没有", "看过什么电影", "看过什么好书",
    "读过什么小说", "遇到什么挑战", "有趣的事", "有没有关注过",
    "你最近在忙什么", "最近在优化哪些具体技术",
    "你最喜欢的音乐类型", "你希望学习哪些具体技能",
    "你感兴趣的具体技能或领域", "你今天想不想", "你猜下次AI会怎么改变我们的生活",
    "你今天想做什么", "你今天的任务是什么", "你今天最想了解什么",
]

# -----------------------------
# 2) 任意位置出现就很危险：整条丢弃
# -----------------------------
DROP_ANYWHERE_KEYWORDS = [
    "minimind",
    "jingyaogong",
    "由jingyaogong创建",
    "由jingyaogong开发",
    "由 Jingyaogong 创造",
    "高效小参数AI模型",
    "阿里云人工智能实验室",
    "阿里云开发的qwen系列",
]

# -----------------------------
# 3) assistant 出现这些：整条丢弃
# -----------------------------
DROP_ASSISTANT_KEYWORDS = [
    # 身份模板 / 训练来源自述
    "我是minimind",
    "我是由jingyaogong",
    "我是由 jingyaogong",
    "我基于开源模型训练",
    "我基于真实数据进行训练",
    "我支持多语言翻译",
    "我支持实时翻译",
    "我的任务是提供准确",
    "我今天将继续提供高效",
    "我专注于多领域知识整合",

    # AI人格 / 能力限制模板
    "我作为ai模型",
    "我没有个人经历",
    "我没有情感",
    "我没有观看电影的能力",
    "我没有阅读书籍的能力",
    "我无法体验",
    "我无法理解和处理",
    "我需要更多上下文信息才能回答这个问题",
    "我无法提供任何美食建议",
]

# -----------------------------
# 4) 元提示 / 串题污染：先删句子，再判定
# -----------------------------
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
    "当然，根据您提供的",
    "如果您需要进一步",
    "那么我该怎么做",
    "非常好，那么请问",
    "请告诉我",
]

# -----------------------------
# 5) 明显串题 / 低质量空话：整条丢弃
# -----------------------------
BAD_ASSISTANT_PATTERNS = [
    r"非常好，那么请问",
    r"请告诉我",
    r"如果您需要进一步",
    r"那么我该怎么做",
    r"当然，根据您提供",
    r"这是一道.*问题.*能不能",
    r"好的，现在我需要",
]

# -----------------------------
# 6) 代码题识别
# -----------------------------
CODE_REQUEST_KEYWORDS = [
    "python", "函数", "代码", "写一个", "编写", "实现", "斐波那契", "质数"
]

CODE_GOOD_MARKERS = [
    "def ", "return", "```python", "```"
]

def contains_think(text: str) -> bool:
    return bool(THINK_PAT.search(text or ""))

def hit_any(text: str, keywords: List[str]) -> bool:
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

    # 统一空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def has_repeated_adjacent_users(convs: List[Dict[str, Any]]) -> bool:
    last_user = None
    for turn in convs:
        if turn.get("role") == "user":
            cur = (turn.get("content") or "").strip()
            if last_user is not None and cur == last_user:
                return True
            last_user = cur
    return False

def is_near_empty_assistant(text: str) -> bool:
    text = (text or "").strip()
    if text in {"", "。", "，", "？", "!", "！"}:
        return True
    if len(text) <= 2:
        return True
    return False

def looks_like_bad_generic_answer(text: str) -> bool:
    """
    过滤那种看似通顺、实际信息密度很低的空泛回答。
    这里只做保守规则，不做太激进的语义判断。
    """
    text = (text or "").strip()

    generic_markers = [
        "提升效率", "提供有意义的信息", "更加全面地理解", "更清晰的分析",
        "通过多人思考", "只有注意问题的解决方案", "促进更高的视角",
        "确保其优点与清晰易懂", "更好地适应新知识",
    ]
    hit_cnt = sum(1 for m in generic_markers if m in text)
    return hit_cnt >= 2

def hits_bad_assistant_pattern(text: str) -> bool:
    text = text or ""
    return any(re.search(p, text) for p in BAD_ASSISTANT_PATTERNS)

def user_requests_code(text: str) -> bool:
    return hit_any(text, CODE_REQUEST_KEYWORDS)

def assistant_has_valid_code_shape(text: str) -> bool:
    text = text or ""
    return any(m in text for m in CODE_GOOD_MARKERS)

def sample_has_bad_code_pair(convs: List[Dict[str, Any]]) -> bool:
    """
    如果 user 明显在要代码，但 assistant 完全没有代码结构，就丢。
    """
    for i in range(len(convs) - 1):
        cur = convs[i]
        nxt = convs[i + 1]
        if cur.get("role") == "user" and nxt.get("role") == "assistant":
            u = cur.get("content", "") or ""
            a = nxt.get("content", "") or ""
            if user_requests_code(u) and not assistant_has_valid_code_shape(a):
                return True
    return False

def should_drop_sample(convs: List[Dict[str, Any]]) -> bool:
    # 至少有 assistant
    if not any(t.get("role") == "assistant" for t in convs):
        return True

    for turn in convs:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        reasoning = (turn.get("reasoning_content") or "").strip()

        # 1) 任意 think 直接丢
        if contains_think(content) or contains_think(reasoning):
            return True

        # 2) 任意位置出现身份模板，直接丢
        if hit_any(content, DROP_ANYWHERE_KEYWORDS) or hit_any(reasoning, DROP_ANYWHERE_KEYWORDS):
            return True

        # 3) user 含身份/人格闲聊问题，直接丢
        if role == "user" and hit_any(content, DROP_USER_KEYWORDS):
            return True

        # 4) assistant 含身份模板/AI人格模板，直接丢
        if role == "assistant" and hit_any(content, DROP_ASSISTANT_KEYWORDS):
            return True

        # 5) 空/近空回答
        if role == "assistant" and is_near_empty_assistant(content):
            return True

        # 6) 编码损坏
        if "�" in content:
            return True

        # 7) 明显串题/元提示残留
        if role == "assistant" and hits_bad_assistant_pattern(content):
            return True

        # 8) 低质量空泛回答
        if role == "assistant" and looks_like_bad_generic_answer(content):
            return True

    # 9) 相邻重复 user
    if has_repeated_adjacent_users(convs):
        return True

    # 10) 坏代码对
    if sample_has_bad_code_pair(convs):
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