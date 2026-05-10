import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import execute_tool_call, get_tool_schemas


CATEGORIES = [
    "hard_multistep_text_math",
    "hard_multistep_math_unit",
    "hard_field_selection",
    "hard_distractor",
    "hard_no_tool",
]

DEFAULT_FULL_SIZES = {
    "hard_eval": 150,
    "hard_sft_train": 800,
    "hard_rl_train": 800,
}

DEFAULT_SMOKE_SIZES = {
    "hard_eval": 31,
    "hard_sft_train": 31,
    "hard_rl_train": 31,
}

TEXT_POOL = [
    "OpenAI API",
    "Agentic RL",
    "MiniMind Agent",
    "tool use eval",
    "abc123",
    "hello world",
    "red blue green",
    "data flywheel",
    "reward model",
    "policy update",
    "function calling",
    "safe rollout",
    "hard eval case",
    "unit test",
    "model output",
    "JSON parser",
    "Shanghai time",
    "math tool",
    "final answer",
    "text length",
]

TEXT_MARKERS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "omega",
    "matrix",
    "vector",
    "signal",
    "anchor",
    "bridge",
    "canvas",
    "driver",
]

TEXT_CONTEXTS = [
    "baseline",
    "sft",
    "rl",
    "holdout",
]

NO_TOOL_TOPICS = [
    "平方根和平方的区别",
    "强化学习的基本目标",
    "什么时候不应该使用工具调用",
    "Agentic RL 的目标",
    "如何制定两步学习计划",
    "如何礼貌地向老师问好",
    "为什么要记录实验结果",
    "什么是 reward hacking",
    "如何避免过度使用工具",
    "为什么要做训练前后对比",
    "什么是 raw 和 guarded 评测",
    "为什么测试要可复现",
    "如何解释函数调用",
    "什么是 observation",
    "为什么要拆分 train 和 eval",
    "如何看待模型幻觉",
    "什么是 SFT",
    "什么是 GRPO",
    "如何做失败样本分析",
    "为什么要保留基础评测",
]

NO_TOOL_STYLES = [
    "用一句话解释",
    "简短说明",
    "面向初学者解释",
    "给出一个清晰定义",
    "请不要计算，直接说明",
    "请用自然语言回答",
    "请给出一个简洁回答",
    "请用两句话以内说明",
    "请用学习笔记风格说明",
    "请礼貌地回答",
]

NO_TOOL_CONTEXTS = [
    "课堂讨论",
    "项目复盘",
    "实验记录",
    "学习笔记",
]

UNIT_PAIRS = [
    ("km", "miles"),
    ("miles", "km"),
    ("kg", "pounds"),
    ("pounds", "kg"),
    ("feet", "meters"),
    ("meters", "feet"),
    ("celsius", "fahrenheit"),
    ("fahrenheit", "celsius"),
]


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_call(name: str, arguments: Dict) -> List[Dict]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def fmt_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def numeric_text(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def split_shift(offset: int) -> int:
    return offset // 10_000


def sample_text(idx: int, offset: int) -> str:
    shift = split_shift(offset)
    marker_idx = idx // len(TEXT_POOL) % len(TEXT_MARKERS)
    return f"{TEXT_POOL[idx % len(TEXT_POOL)]} {TEXT_MARKERS[marker_idx]} {TEXT_CONTEXTS[shift % len(TEXT_CONTEXTS)]}"


def eval_item(
    task_id: str,
    category: str,
    prompt: str,
    tools: List[str],
    expected_sequence: List[str],
    expected_args=None,
    expected_answer=None,
    expected_obs_keys=None,
    args_match: str = "equivalent",
    max_tool_calls: int = 1,
    allow_no_tool: bool = False,
) -> Dict:
    return {
        "id": task_id,
        "category": category,
        "prompt": prompt,
        "tools": tools,
        "expected_tool_sequence": expected_sequence,
        "expected_args": expected_args or {},
        "expected_obs_keys": expected_obs_keys or {},
        "args_match": args_match,
        "expected_answer": expected_answer,
        "expected_answer_type": "none" if expected_answer is None else "number",
        "max_tool_calls": max_tool_calls,
        "allow_no_tool": allow_no_tool,
    }


def sft_no_tool_with_tools(prompt: str, tools: List[str], final_answer: str) -> Dict:
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tools), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer},
        ]
    }


def sft_multi_tool_sample(prompt: str, tools: List[str], steps: List[Tuple[str, Dict]], final_answer: str) -> Dict:
    conversations = [
        {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tools), ensure_ascii=False)},
        {"role": "user", "content": prompt},
    ]
    for name, arguments in steps:
        result = execute_tool_call(name, arguments)
        conversations.append({"role": "assistant", "content": "", "tool_calls": json.dumps(tool_call(name, arguments), ensure_ascii=False)})
        conversations.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    conversations.append({"role": "assistant", "content": final_answer})
    return {"conversations": conversations}


def allocate_counts(total: int) -> Dict[str, int]:
    if total < len(CATEGORIES):
        raise ValueError(f"total must be at least {len(CATEGORIES)}")
    base = total // len(CATEGORIES)
    remainder = total % len(CATEGORIES)
    return {category: base + (1 if i < remainder else 0) for i, category in enumerate(CATEGORIES)}


def text_stats(text: str) -> Dict:
    return execute_tool_call("text_length", {"text": text})


def math_result(expression: str):
    return execute_tool_call("calculate_math", {"expression": expression})["result"]


def make_text_math_case(split: str, idx: int, offset: int) -> Tuple[Dict, Dict]:
    text = sample_text(idx, offset)
    stats = text_stats(text)
    field = "characters" if (idx + offset) % 2 == 0 else "words"
    base_value = int(stats[field])
    op_kind = (idx + offset) % 5
    k = (idx + offset) % 9 + 2
    if op_kind == 0:
        expression = f"{base_value}*{k}"
        phrase = f"乘以{k}"
    elif op_kind == 1:
        expression = f"{base_value}+{k}"
        phrase = f"加上{k}"
    elif op_kind == 2:
        expression = f"{base_value}-{min(k, max(base_value - 1, 1))}"
        phrase = f"减去{min(k, max(base_value - 1, 1))}"
    elif op_kind == 3:
        divisor = 2 if base_value % 2 == 0 else 1
        expression = f"{base_value}/{divisor}"
        phrase = f"除以{divisor}"
    else:
        expression = f"{base_value}**2"
        phrase = "计算它的平方"
    answer = numeric_text(math_result(expression))
    field_zh = "字符数" if field == "characters" else "单词数"
    prompt = f"统计 {text} 的{field_zh}，然后{phrase}"
    tools = ["text_length", "calculate_math", "unit_converter"]
    steps = [("text_length", {"text": text}), ("calculate_math", {"expression": expression})]
    row = eval_item(
        f"{split}_tm_{idx + 1:04d}",
        "hard_multistep_text_math",
        prompt,
        tools,
        ["text_length", "calculate_math"],
        {"text_length": {"text": text}, "calculate_math": {"expression": expression}},
        answer,
        {"text_length": [field], "calculate_math": ["result"]},
        max_tool_calls=2,
    )
    sft = sft_multi_tool_sample(prompt, tools, steps, f"计算结果是 {answer}。")
    return row, sft


def expression_for_index(idx: int, offset: int) -> Tuple[str, float]:
    shift = split_shift(offset)
    a = idx + 3 + shift * 200
    b = (idx * 3) % 19 + 2 + shift * 30
    kind = idx % 4
    if kind == 0:
        expression = f"{a}+{b}"
    elif kind == 1:
        expression = f"{a}*{b}"
    elif kind == 2:
        expression = f"{a + b}-{b}"
    else:
        root = idx + 2 + shift * 100
        expression = f"sqrt({root * root})"
    return expression, math_result(expression)


def make_math_unit_case(split: str, idx: int, offset: int) -> Tuple[Dict, Dict]:
    expression, value = expression_for_index(idx, offset)
    from_unit, to_unit = UNIT_PAIRS[(idx + offset) % len(UNIT_PAIRS)]
    unit_args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
    unit_result = execute_tool_call("unit_converter", unit_args)
    answer = numeric_text(unit_result["result"])
    prompt = f"先计算 {expression}，再把这个数值的 {from_unit} 换算成 {to_unit}"
    tools = ["calculate_math", "unit_converter", "text_length"]
    steps = [("calculate_math", {"expression": expression}), ("unit_converter", unit_args)]
    row = eval_item(
        f"{split}_mu_{idx + 1:04d}",
        "hard_multistep_math_unit",
        prompt,
        tools,
        ["calculate_math", "unit_converter"],
        {"calculate_math": {"expression": expression}, "unit_converter": unit_args},
        answer,
        {"calculate_math": ["result"], "unit_converter": ["result"]},
        max_tool_calls=2,
    )
    sft = sft_multi_tool_sample(prompt, tools, steps, f"{fmt_number(value)} {from_unit} = {answer} {to_unit}。")
    return row, sft


def make_field_selection_case(split: str, idx: int, offset: int) -> Tuple[Dict, Dict]:
    if (idx + offset) % 2 == 0:
        text = sample_text(idx, offset)
        field = "characters" if (idx + offset) % 4 in {0, 1} else "words"
        stats = text_stats(text)
        answer = numeric_text(stats[field])
        forbidden = "单词数" if field == "characters" else "字符数"
        target = "字符数" if field == "characters" else "单词数"
        prompt = f"统计 {text} 的{target}，不要回答{forbidden}"
        tools = ["text_length", "calculate_math"]
        args = {"text": text}
        row = eval_item(
            f"{split}_field_{idx + 1:04d}",
            "hard_field_selection",
            prompt,
            tools,
            ["text_length"],
            {"text_length": args},
            answer,
            {"text_length": [field]},
        )
        unit = "字符" if field == "characters" else "单词"
        sft = sft_multi_tool_sample(prompt, tools, [("text_length", args)], f"{text} 有 {answer} 个{unit}。")
        return row, sft

    value = idx % 25 + 1 + split_shift(offset) * 30
    from_unit, to_unit = UNIT_PAIRS[(idx + offset) % len(UNIT_PAIRS)]
    args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
    result = execute_tool_call("unit_converter", args)
    answer = numeric_text(result["result"])
    prompt = f"把 {value} {from_unit} 换算成 {to_unit}，只回答换算后的 result"
    tools = ["unit_converter", "calculate_math"]
    row = eval_item(
        f"{split}_field_{idx + 1:04d}",
        "hard_field_selection",
        prompt,
        tools,
        ["unit_converter"],
        {"unit_converter": args},
        answer,
        {"unit_converter": ["result"]},
    )
    sft = sft_multi_tool_sample(prompt, tools, [("unit_converter", args)], f"{fmt_number(value)} {from_unit} = {answer} {to_unit}。")
    return row, sft


def make_distractor_case(split: str, idx: int, offset: int) -> Tuple[Dict, Dict]:
    kind = (idx + offset) % 3
    if kind == 0:
        shift = split_shift(offset)
        a = idx % 80 + 11 + shift * 100
        b = (idx * 7) % 80 + 12 + shift * 100
        expression = f"{a}+{b}"
        answer = numeric_text(math_result(expression))
        prompt = f"不要查询时间也不要随机生成，直接计算 {expression}"
        tools = ["get_current_time", "random_number", "calculate_math"]
        name, args = "calculate_math", {"expression": expression}
        expected_obs = {"calculate_math": ["result"]}
        final_answer = f"{expression} = {answer}。"
    elif kind == 1:
        text = sample_text(idx, offset)
        field = "words" if (idx + offset) % 2 == 0 else "characters"
        stats = text_stats(text)
        answer = numeric_text(stats[field])
        target = "单词数" if field == "words" else "字符数"
        prompt = f"不要调用计算器，统计 {text} 的{target}"
        tools = ["text_length", "calculate_math"]
        name, args = "text_length", {"text": text}
        expected_obs = {"text_length": [field]}
        final_answer = f"这段文本共有 {answer} 个{'单词' if field == 'words' else '字符'}。"
    else:
        value = idx + 2 + split_shift(offset) * 30
        from_unit, to_unit = UNIT_PAIRS[(idx + offset) % len(UNIT_PAIRS)]
        args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
        result = execute_tool_call("unit_converter", args)
        answer = numeric_text(result["result"])
        prompt = f"虽然有计算器，但请把 {value} {from_unit} 换算成 {to_unit}"
        tools = ["unit_converter", "calculate_math"]
        name = "unit_converter"
        expected_obs = {"unit_converter": ["result"]}
        final_answer = f"{value} {from_unit} = {answer} {to_unit}。"

    row = eval_item(
        f"{split}_dist_{idx + 1:04d}",
        "hard_distractor",
        prompt,
        tools,
        [name],
        {name: args},
        answer,
        expected_obs,
    )
    sft = sft_multi_tool_sample(prompt, tools, [(name, args)], final_answer)
    return row, sft


def make_no_tool_case(split: str, idx: int, offset: int) -> Tuple[Dict, Dict]:
    shift = split_shift(offset)
    topic = NO_TOOL_TOPICS[idx % len(NO_TOOL_TOPICS)]
    style = NO_TOOL_STYLES[(idx // len(NO_TOOL_TOPICS) + shift * 3) % len(NO_TOOL_STYLES)]
    limit = 20 + ((idx + shift * 2) % 7) * 5
    context = NO_TOOL_CONTEXTS[shift % len(NO_TOOL_CONTEXTS)]
    prompt = f"{style}：{topic}，用于{context}，不超过{limit}个字"
    tools = ["calculate_math", "text_length", "get_current_time"]
    answer = f"{topic}可以用自然语言说明，不需要调用工具。"
    row = eval_item(
        f"{split}_notool_{idx + 1:04d}",
        "hard_no_tool",
        prompt,
        tools,
        [],
        {},
        None,
        {},
        args_match="none",
        max_tool_calls=0,
        allow_no_tool=True,
    )
    return row, sft_no_tool_with_tools(prompt, tools, answer)


CASE_BUILDERS = {
    "hard_multistep_text_math": make_text_math_case,
    "hard_multistep_math_unit": make_math_unit_case,
    "hard_field_selection": make_field_selection_case,
    "hard_distractor": make_distractor_case,
    "hard_no_tool": make_no_tool_case,
}


def build_split_rows(split: str, total_rows: int, offset: int) -> Tuple[List[Dict], List[Dict]]:
    counts = allocate_counts(total_rows)
    eval_rows: List[Dict] = []
    sft_rows: List[Dict] = []
    for category in CATEGORIES:
        builder = CASE_BUILDERS[category]
        for idx in range(counts[category]):
            row, sft = builder(split, idx, offset)
            eval_rows.append(row)
            sft_rows.append(sft)
    validate_rows(eval_rows, total_rows, split)
    return eval_rows, sft_rows


def build_rl_rows(eval_rows: Iterable[Dict]) -> List[Dict]:
    rows = []
    for item in eval_rows:
        row = dict(item)
        row["gt"] = [] if item.get("expected_answer") is None else [str(item["expected_answer"])]
        rows.append(row)
    return rows


def validate_rows(rows: Sequence[Dict], expected_count: int, label: str) -> None:
    ids = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(rows) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} rows, got {len(rows)}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate ids detected")
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"{label}: duplicate prompts detected")
    categories = {row["category"] for row in rows}
    missing = [category for category in CATEGORIES if category not in categories]
    if missing:
        raise ValueError(f"{label}: missing categories {missing}")


def validate_no_prompt_overlap(left: Sequence[Dict], right: Sequence[Dict], left_name: str, right_name: str) -> None:
    overlap = {row["prompt"] for row in left} & {row["prompt"] for row in right}
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(f"{left_name}/{right_name}: prompt overlap detected: {sample}")


def build_all_splits(
    hard_eval_size: int = DEFAULT_FULL_SIZES["hard_eval"],
    hard_sft_train_size: int = DEFAULT_FULL_SIZES["hard_sft_train"],
    hard_rl_train_size: int = DEFAULT_FULL_SIZES["hard_rl_train"],
) -> Dict[str, List[Dict]]:
    hard_eval, _ = build_split_rows("hard_eval", hard_eval_size, offset=0)
    sft_eval_like, hard_sft_train = build_split_rows("hard_sft_train", hard_sft_train_size, offset=10_000)
    rl_eval_like, _ = build_split_rows("hard_rl_train", hard_rl_train_size, offset=20_000)
    hard_rl_train = build_rl_rows(rl_eval_like)

    validate_no_prompt_overlap(hard_eval, sft_eval_like, "hard_eval", "hard_sft_train")
    validate_no_prompt_overlap(hard_eval, rl_eval_like, "hard_eval", "hard_rl_train")
    validate_no_prompt_overlap(sft_eval_like, rl_eval_like, "hard_sft_train", "hard_rl_train")

    return {
        "hard_eval": hard_eval,
        "hard_sft_train": hard_sft_train,
        "hard_rl_train": hard_rl_train,
        "_hard_sft_eval_like": sft_eval_like,
        "_hard_rl_eval_like": rl_eval_like,
    }


def summarize(rows: Sequence[Dict]) -> Dict[str, int]:
    summary = {category: 0 for category in CATEGORIES}
    for row in rows:
        summary[row["category"]] += 1
    return summary


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic hard Tool-Use Agentic RL data")
    parser.add_argument("--preset", choices=["full", "smoke"], default="full")
    parser.add_argument("--hard_eval_size", type=int, default=0)
    parser.add_argument("--hard_sft_train_size", type=int, default=0)
    parser.add_argument("--hard_rl_train_size", type=int, default=0)
    parser.add_argument("--eval_path", default="evals/tool_eval_hard.jsonl")
    parser.add_argument("--sft_path", default="dataset/tool_sft_hard_train.jsonl")
    parser.add_argument("--rl_path", default="dataset/agent_rl_tooluse_hard_train.jsonl")
    parser.add_argument("--dry_run", action="store_true", help="只构造和校验数据，不写入文件")
    args = parser.parse_args()

    defaults = DEFAULT_SMOKE_SIZES if args.preset == "smoke" else DEFAULT_FULL_SIZES
    hard_eval_size = args.hard_eval_size or defaults["hard_eval"]
    hard_sft_train_size = args.hard_sft_train_size or defaults["hard_sft_train"]
    hard_rl_train_size = args.hard_rl_train_size or defaults["hard_rl_train"]

    splits = build_all_splits(
        hard_eval_size=hard_eval_size,
        hard_sft_train_size=hard_sft_train_size,
        hard_rl_train_size=hard_rl_train_size,
    )

    print(f"hard eval rows: {len(splits['hard_eval'])} {summarize(splits['hard_eval'])}")
    print(f"hard sft train rows: {len(splits['hard_sft_train'])}")
    print(f"hard rl train rows: {len(splits['hard_rl_train'])} {summarize(splits['_hard_rl_eval_like'])}")
    print("prompt overlap: none")

    if args.dry_run:
        print("dry run: no files written")
        return

    eval_path = resolve_path(args.eval_path)
    sft_path = resolve_path(args.sft_path)
    rl_path = resolve_path(args.rl_path)
    write_jsonl(eval_path, splits["hard_eval"])
    write_jsonl(sft_path, splits["hard_sft_train"])
    write_jsonl(rl_path, splits["hard_rl_train"])
    (PROJECT_ROOT / "evals" / "reports").mkdir(parents=True, exist_ok=True)

    print(f"wrote hard eval -> {eval_path}")
    print(f"wrote hard sft train -> {sft_path}")
    print(f"wrote hard rl train -> {rl_path}")


if __name__ == "__main__":
    main()
