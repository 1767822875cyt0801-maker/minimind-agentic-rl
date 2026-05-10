import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import execute_tool_call, get_tool_schemas


DYNAMIC_CATEGORIES = [
    "dynamic_text_math_unit",
    "dynamic_math_unit_math",
    "dynamic_strong_distractor",
    "dynamic_no_tool",
]
LANGUAGE_CHOICES = ("zh", "en", "mixed")

DEFAULT_EVAL_SIZE = 80
DEFAULT_TRAIN_SIZE = 800
DEFAULT_SFT_SEED_SIZE = 320
SMOKE_EVAL_SIZE = 24
SMOKE_TRAIN_SIZE = 48
SMOKE_SFT_SEED_SIZE = 48

EN_TEXT_POOL = [
    "OpenAI API benchmark",
    "MiniMind tool reward",
    "Agentic RL rollout",
    "dynamic verifier trace",
    "function call parser",
    "guarded environment",
    "observation grounded answer",
    "reward variance signal",
]
ZH_TEXT_POOL = [
    "OpenAI API 基准测试",
    "MiniMind 工具奖励",
    "Agentic RL 轨迹记录",
    "动态验证器轨迹",
    "函数调用解析器",
    "受控工具环境",
    "基于观察的答案",
    "奖励方差信号",
]
TEXT_POOLS = {"en": EN_TEXT_POOL, "zh": ZH_TEXT_POOL}

UNIT_PAIRS = [
    ("km", "miles"),
    ("miles", "km"),
    ("kg", "pounds"),
    ("pounds", "kg"),
    ("meters", "feet"),
    ("feet", "meters"),
]

EN_NO_TOOL_TOPICS = [
    "why experiments need a held-out eval split",
    "how to avoid overusing tools",
    "what reward variance means in RL",
    "why a short final answer can be better",
    "when a model should answer without tools",
    "how to analyze failure counts",
]
ZH_NO_TOOL_TOPICS = [
    "为什么实验需要保留评测集",
    "如何避免过度调用工具",
    "什么是奖励方差",
    "为什么简短最终答案更容易验证",
    "什么时候模型应该不调用工具直接回答",
    "如何分析失败计数",
]
NO_TOOL_TOPICS = {"en": EN_NO_TOOL_TOPICS, "zh": ZH_NO_TOOL_TOPICS}


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_call(name: str, arguments: Dict) -> List[Dict]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def fmt_number(value) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def text_stats(text: str) -> Dict[str, int]:
    return {"characters": len(text), "words": len(text.split())}


def choose_language(language: str, level: int, idx: int, category_idx: int = 0) -> str:
    if language == "mixed":
        return "zh" if (level + idx + category_idx) % 2 == 0 else "en"
    if language not in {"zh", "en"}:
        raise ValueError(f"language must be one of {LANGUAGE_CHOICES}, got {language}")
    return language


def sample_text(language: str, level: int, idx: int, offset: int) -> str:
    pool = TEXT_POOLS[language]
    base = pool[(idx + offset) % len(pool)]
    if language == "zh":
        return f"{base} 第{level}级 样本 {idx + offset}"
    return f"{base} level {level} case {idx + offset}"


def field_name(field: str, language: str) -> str:
    if language == "zh":
        return "字符数" if field == "characters" else "单词数"
    return field


def tool_result(name: str, args: Dict) -> Dict:
    return execute_tool_call(name, args)


def numeric_result(name: str, args: Dict) -> float:
    result = tool_result(name, args)
    return float(result["result"])


def eval_item(
    task_id: str,
    category: str,
    level: int,
    prompt: str,
    tools: List[str],
    expected_sequence: List[str],
    expected_args,
    expected_obs_keys,
    expected_answer,
    max_tool_calls: int,
    allow_no_tool: bool = False,
    language: str = "zh",
) -> Dict:
    return {
        "id": task_id,
        "category": category,
        "difficulty_level": level,
        "language": language,
        "prompt": prompt,
        "tools": tools,
        "expected_tool_sequence": expected_sequence,
        "expected_args": expected_args,
        "expected_obs_keys": expected_obs_keys,
        "expected_answer": expected_answer,
        "expected_answer_type": "none" if expected_answer is None else "number",
        "gt": [] if expected_answer is None else [str(expected_answer)],
        "max_tool_calls": max_tool_calls,
        "allow_no_tool": allow_no_tool,
        "reward_profile": "dense_v2",
    }


def make_text_math_unit_case(split: str, level: int, idx: int, offset: int, language: str = "zh") -> Dict:
    text = sample_text(language, level, idx, offset)
    field = "characters" if (idx + level + offset) % 2 == 0 else "words"
    base_value = text_stats(text)[field]
    multiplier = level + 2 + (idx % 4)
    addend = (idx * 3 + level) % 17 + 1
    expression = f"{base_value}*{multiplier}+{addend}"
    math_value = numeric_result("calculate_math", {"expression": expression})
    from_unit, to_unit = UNIT_PAIRS[(idx + level + offset) % len(UNIT_PAIRS)]
    unit_args = {"value": math_value, "from_unit": from_unit, "to_unit": to_unit}
    answer = fmt_number(numeric_result("unit_converter", unit_args))
    if language == "zh":
        prompt = (
            f"统计「{text}」的{field_name(field, language)}，把这个数乘以 {multiplier}，再加 {addend}，"
            f"然后从 {from_unit} 换算成 {to_unit}；忽略干扰数字 {idx + 91}。"
        )
    else:
        prompt = (
            f"Count the {field} in '{text}', multiply that count by {multiplier}, "
            f"add {addend}, then convert the result from {from_unit} to {to_unit}. "
            f"Ignore the distractor number {idx + 91}."
        )
    return eval_item(
        f"{split}_l{level}_tmu_{idx + 1:04d}",
        "dynamic_text_math_unit",
        level,
        prompt,
        ["text_length", "calculate_math", "unit_converter", "get_current_time"],
        ["text_length", "calculate_math", "unit_converter"],
        {
            "text_length": {"text": text},
            "calculate_math": {"expression": expression},
            "unit_converter": unit_args,
        },
        {"text_length": [field], "calculate_math": ["result"], "unit_converter": ["result"]},
        answer,
        max_tool_calls=3,
        language=language,
    )


def make_math_unit_math_case(split: str, level: int, idx: int, offset: int, language: str = "zh") -> Dict:
    a = 17 + idx + level * 11 + offset
    b = 3 + (idx * 5) % 19
    first_expression = f"{a}*{b}-{level + idx % 7}"
    first_value = numeric_result("calculate_math", {"expression": first_expression})
    from_unit, to_unit = UNIT_PAIRS[(idx * 2 + level + offset) % len(UNIT_PAIRS)]
    unit_args = {"value": first_value, "from_unit": from_unit, "to_unit": to_unit}
    converted = numeric_result("unit_converter", unit_args)
    divisor = level + 1
    addend = (idx % 9) + level
    second_expression = f"{fmt_number(converted)}/{divisor}+{addend}"
    answer = fmt_number(numeric_result("calculate_math", {"expression": second_expression}))
    if language == "zh":
        prompt = (
            f"先计算 {first_expression}。把这个结果从 {from_unit} 换算成 {to_unit}。"
            f"最后把换算结果除以 {divisor} 并加 {addend}。只回答最终数字。"
        )
    else:
        prompt = (
            f"First compute {first_expression}. Convert that value from {from_unit} to {to_unit}. "
            f"Then divide the converted result by {divisor} and add {addend}. "
            f"Only the final number is needed."
        )
    return eval_item(
        f"{split}_l{level}_mum_{idx + 1:04d}",
        "dynamic_math_unit_math",
        level,
        prompt,
        ["calculate_math", "unit_converter", "text_length"],
        ["calculate_math", "unit_converter", "calculate_math"],
        {
            "calculate_math": [
                {"expression": first_expression},
                {"expression": second_expression},
            ],
            "unit_converter": unit_args,
        },
        {"calculate_math": ["result"], "unit_converter": ["result"]},
        answer,
        max_tool_calls=3,
        language=language,
    )


def make_strong_distractor_case(split: str, level: int, idx: int, offset: int, language: str = "zh") -> Dict:
    text = sample_text(language, level, idx * 3, offset)
    if language == "zh":
        text = f"{text} 干扰项 {idx + level}"
    else:
        text = f"{text} distractor {idx + level} split {offset}"
    stats = text_stats(text)
    wrong_number = stats["words"] + 100 + level
    expression = f"{stats['characters']}+{level * 7 + idx % 5}"
    answer = fmt_number(numeric_result("calculate_math", {"expression": expression}))
    if language == "zh":
        prompt = (
            f"不要查询时间，也不要随机生成；数字 {wrong_number} 是干扰项。"
            f"统计「{text}」的字符数，加上 {level * 7 + idx % 5}，并回答结果。"
        )
    else:
        prompt = (
            f"Do not use time or random tools. The number {wrong_number} is irrelevant. "
            f"Use the character count of '{text}', add {level * 7 + idx % 5}, and answer the result."
        )
    return eval_item(
        f"{split}_l{level}_dist_{idx + 1:04d}",
        "dynamic_strong_distractor",
        level,
        prompt,
        ["get_current_time", "random_number", "text_length", "calculate_math"],
        ["text_length", "calculate_math"],
        {
            "text_length": {"text": text},
            "calculate_math": {"expression": expression},
        },
        {"text_length": ["characters"], "calculate_math": ["result"]},
        answer,
        max_tool_calls=2,
        language=language,
    )


def make_no_tool_case(split: str, level: int, idx: int, offset: int, language: str = "zh") -> Dict:
    topics = NO_TOOL_TOPICS[language]
    topic = topics[(idx + offset + level) % len(topics)]
    if language == "zh":
        prompt = f"不调用任何工具，用一句简短的话回答：{topic}。数字 {idx + 31 + offset} 和 {level * 13} 是干扰项。"
    else:
        prompt = (
            f"Answer in one short sentence without calling any tool: {topic}. "
            f"The numbers {idx + 31 + offset} and {level * 13} are distractors."
        )
    return eval_item(
        f"{split}_l{level}_notool_{idx + 1:04d}",
        "dynamic_no_tool",
        level,
        prompt,
        ["calculate_math", "text_length", "get_current_time"],
        [],
        {},
        {},
        None,
        max_tool_calls=0,
        allow_no_tool=True,
        language=language,
    )


BUILDERS = [
    make_text_math_unit_case,
    make_math_unit_math_case,
    make_strong_distractor_case,
    make_no_tool_case,
]


def allocate_counts(total: int) -> Dict[str, int]:
    if total < len(DYNAMIC_CATEGORIES):
        raise ValueError(f"total must be at least {len(DYNAMIC_CATEGORIES)}")
    base = total // len(DYNAMIC_CATEGORIES)
    rem = total % len(DYNAMIC_CATEGORIES)
    return {cat: base + (1 if i < rem else 0) for i, cat in enumerate(DYNAMIC_CATEGORIES)}


def build_level_rows(split: str, level: int, total: int, offset: int, language: str = "zh") -> List[Dict]:
    counts = allocate_counts(total)
    rows = []
    for category_idx, (category, builder) in enumerate(zip(DYNAMIC_CATEGORIES, BUILDERS)):
        for idx in range(counts[category]):
            row_language = choose_language(language, level, idx, category_idx)
            rows.append(builder(split, level, idx, offset, row_language))
    validate_rows(rows, total, f"{split}_l{level}")
    return rows


def build_dynamic_train(levels: Iterable[int], total: int, offset: int, language: str = "zh") -> List[Dict]:
    levels = list(levels)
    if not levels:
        raise ValueError("levels must not be empty")
    per_level = total // len(levels)
    remainder = total % len(levels)
    rows = []
    for i, level in enumerate(levels):
        rows.extend(
            build_level_rows(
                "dynamic_train",
                level,
                per_level + (1 if i < remainder else 0),
                offset + level * 10000,
                language=language,
            )
        )
    validate_rows(rows, total, "dynamic_train")
    return rows


def expected_args_for_step(row: Dict, tool_name: str, occurrence_idx: int) -> Dict:
    value = row.get("expected_args", {}).get(tool_name, {})
    if isinstance(value, list):
        if occurrence_idx < len(value):
            return value[occurrence_idx]
        return {}
    return value if isinstance(value, dict) else {}


def direct_final_answer(prompt: str, language: str = "zh") -> str:
    if language == "zh":
        if "保留评测集" in prompt:
            return "保留评测集可以检查模型是否真的泛化。"
        if "过度调用工具" in prompt:
            return "能直接回答时少用工具，只在需要计算或查询时调用。"
        if "奖励方差" in prompt:
            return "奖励方差表示不同回答之间是否有可学习的差异。"
        if "简短最终答案" in prompt:
            return "简短最终答案更容易核验，也更不容易跑偏。"
        if "不调用工具" in prompt or "直接回答" in prompt:
            return "不需要计算、查询或结构化结果时应直接回答。"
        if "失败计数" in prompt:
            return "失败计数能帮助定位最需要修复的行为。"
        return "这个问题可以直接回答，不需要调用工具。"
    if "held-out eval split" in prompt:
        return "A held-out eval split checks generalization without training leakage."
    if "overusing tools" in prompt:
        return "Avoid tools when the answer needs explanation rather than exact external computation."
    if "reward variance" in prompt:
        return "Reward variance shows whether RL has useful preference signal."
    if "short final answer" in prompt:
        return "A short final answer is easier to verify and less likely to drift."
    if "answer without tools" in prompt:
        return "Answer directly when no calculation, lookup, or structured tool result is needed."
    if "failure counts" in prompt:
        return "Failure counts reveal which behavior should be repaired first."
    return "This can be answered directly without using a tool."


def sft_sample_from_task(row: Dict) -> Dict:
    conversations = [
        {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(row["tools"]), ensure_ascii=False)},
        {"role": "user", "content": row["prompt"]},
    ]
    occurrences: Dict[str, int] = {}
    for name in row["expected_tool_sequence"]:
        occurrence_idx = occurrences.get(name, 0)
        arguments = expected_args_for_step(row, name, occurrence_idx)
        occurrences[name] = occurrence_idx + 1
        result = execute_tool_call(name, arguments)
        conversations.append(
            {"role": "assistant", "content": "", "tool_calls": json.dumps(tool_call(name, arguments), ensure_ascii=False)}
        )
        conversations.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    language = row.get("language", "zh")
    if row.get("allow_no_tool"):
        final_answer = direct_final_answer(row["prompt"], language=language)
    elif language == "zh":
        final_answer = f"最终答案是 {row['expected_answer']}。"
    else:
        final_answer = f"The final answer is {row['expected_answer']}."
    conversations.append({"role": "assistant", "content": final_answer})
    return {
        "id": f"{row['id']}_sft",
        "category": row.get("category"),
        "difficulty_level": row.get("difficulty_level"),
        "language": language,
        "conversations": conversations,
    }


def build_dynamic_sft_seed(levels: Iterable[int], total: int, offset: int, language: str = "zh") -> List[Dict]:
    tasks = build_dynamic_train(levels, total, offset, language=language)
    return [sft_sample_from_task(row) for row in tasks]


def parse_levels(value: str) -> List[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        level = int(part)
        if level < 1 or level > 4:
            raise ValueError(f"level must be in 1..4, got {level}")
        levels.append(level)
    if not levels:
        raise ValueError("at least one level is required")
    return levels


def validate_rows(rows: Sequence[Dict], expected_count: int, label: str) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{label}: expected {expected_count}, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate ids")
    prompts = [row["prompt"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"{label}: duplicate prompts")
    for row in rows:
        for key in ["gt", "expected_answer", "difficulty_level", "language", "reward_profile"]:
            if key not in row:
                raise ValueError(f"{label}: missing {key} in {row.get('id')}")
        if row["language"] not in {"zh", "en"}:
            raise ValueError(f"{label}: invalid language {row['language']} in {row.get('id')}")


def validate_no_prompt_overlap(named_rows: Dict[str, Sequence[Dict]]) -> None:
    seen: Dict[str, str] = {}
    for name, rows in named_rows.items():
        for row in rows:
            prompt = row_prompt(row)
            if prompt in seen:
                raise ValueError(f"prompt overlap: {seen[prompt]} and {name}: {prompt}")
            seen[prompt] = name


def row_prompt(row: Dict) -> str:
    if "prompt" in row:
        return row["prompt"]
    for message in row.get("conversations", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def summarize(rows: Sequence[Dict]) -> Dict[str, int]:
    counts = {cat: 0 for cat in DYNAMIC_CATEGORIES}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def build_all_splits(
    eval_size: int = DEFAULT_EVAL_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    sft_seed_size: int = DEFAULT_SFT_SEED_SIZE,
    train_levels: Iterable[int] = (3, 4),
    sft_seed_levels: Iterable[int] = (1, 2, 3, 4),
    language: str = "zh",
) -> Dict[str, List[Dict]]:
    splits = {
        "l1": build_level_rows("dynamic_eval", 1, eval_size, 1000, language=language),
        "l2": build_level_rows("dynamic_eval", 2, eval_size, 2000, language=language),
        "l3": build_level_rows("dynamic_eval", 3, eval_size, 3000, language=language),
        "l4": build_level_rows("dynamic_eval", 4, eval_size, 4000, language=language),
        "train": build_dynamic_train(train_levels, train_size, 20000, language=language),
        "sft_seed": build_dynamic_sft_seed(sft_seed_levels, sft_seed_size, 50000, language=language),
    }
    validate_no_prompt_overlap(splits)
    return splits


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def default_output_paths(language: str) -> Dict[str, str]:
    if language == "en":
        prefix = "dynamic"
        return {
            "l1_path": f"evals/tool_eval_{prefix}_l1.jsonl",
            "l2_path": f"evals/tool_eval_{prefix}_l2.jsonl",
            "l3_path": f"evals/tool_eval_{prefix}_l3.jsonl",
            "l4_path": f"evals/tool_eval_{prefix}_l4.jsonl",
            "train_path": "dataset/agent_rl_tooluse_dynamic_train.jsonl",
            "sft_seed_path": "dataset/tool_sft_dynamic_seed.jsonl",
        }
    if language == "zh":
        prefix = "dynamic_zh"
    elif language == "mixed":
        prefix = "dynamic_mixed"
    else:
        raise ValueError(f"language must be one of {LANGUAGE_CHOICES}, got {language}")
    return {
        "l1_path": f"evals/tool_eval_{prefix}_l1.jsonl",
        "l2_path": f"evals/tool_eval_{prefix}_l2.jsonl",
        "l3_path": f"evals/tool_eval_{prefix}_l3.jsonl",
        "l4_path": f"evals/tool_eval_{prefix}_l4.jsonl",
        "train_path": f"dataset/agent_rl_tooluse_{prefix}_train.jsonl",
        "sft_seed_path": f"dataset/tool_sft_{prefix}_seed.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dynamic tool-use RLVR data")
    parser.add_argument("--preset", choices=["full", "smoke"], default="full")
    parser.add_argument("--language", choices=LANGUAGE_CHOICES, default="zh")
    parser.add_argument("--eval_size", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--sft_seed_size", type=int, default=0)
    parser.add_argument("--train_levels", default="3,4")
    parser.add_argument("--sft_seed_levels", default="1,2,3,4")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--l1_path", default=None)
    parser.add_argument("--l2_path", default=None)
    parser.add_argument("--l3_path", default=None)
    parser.add_argument("--l4_path", default=None)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--sft_seed_path", default=None)
    args = parser.parse_args()

    eval_size = args.eval_size or (SMOKE_EVAL_SIZE if args.preset == "smoke" else DEFAULT_EVAL_SIZE)
    train_size = args.train_size or (SMOKE_TRAIN_SIZE if args.preset == "smoke" else DEFAULT_TRAIN_SIZE)
    sft_seed_size = args.sft_seed_size or (SMOKE_SFT_SEED_SIZE if args.preset == "smoke" else DEFAULT_SFT_SEED_SIZE)
    train_levels = parse_levels(args.train_levels)
    sft_seed_levels = parse_levels(args.sft_seed_levels)
    splits = build_all_splits(
        eval_size=eval_size,
        train_size=train_size,
        sft_seed_size=sft_seed_size,
        train_levels=train_levels,
        sft_seed_levels=sft_seed_levels,
        language=args.language,
    )

    print(f"language: {args.language}")
    for key in ["l1", "l2", "l3", "l4"]:
        print(f"dynamic {key} rows: {len(splits[key])} {summarize(splits[key])}")
    print(f"dynamic train rows: {len(splits['train'])} levels={train_levels} {summarize(splits['train'])}")
    print(f"dynamic sft seed rows: {len(splits['sft_seed'])} levels={sft_seed_levels}")
    print("prompt overlap: none")

    if args.dry_run:
        print("dry run: no files written")
        return

    defaults = default_output_paths(args.language)
    outputs = {
        args.l1_path or defaults["l1_path"]: splits["l1"],
        args.l2_path or defaults["l2_path"]: splits["l2"],
        args.l3_path or defaults["l3_path"]: splits["l3"],
        args.l4_path or defaults["l4_path"]: splits["l4"],
        args.train_path or defaults["train_path"]: splits["train"],
        args.sft_seed_path or defaults["sft_seed_path"]: splits["sft_seed"],
    }
    for path, rows in outputs.items():
        target = resolve_path(path)
        write_jsonl(target, rows)
        print(f"wrote {len(rows)} rows -> {target}")


if __name__ == "__main__":
    main()
