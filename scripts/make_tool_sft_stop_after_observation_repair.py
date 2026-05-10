import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import execute_tool_call, get_tool_schemas


DEFAULT_ROWS = 1200

TEXTS = [
    "alpha beta",
    "MiniMind RL",
    "tool call parser",
    "safe rollout trace",
    "OpenAI API",
    "reward variance",
    "function calling",
    "agent loop guard",
    "json schema",
    "final answer",
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

NO_TOOL_TOPICS = [
    (
        "Explain what a tool call is in one sentence.",
        "A tool call asks an external function to do a precise operation and return an observation.",
    ),
    (
        "Say hello politely without using any tool.",
        "Hello, nice to meet you.",
    ),
    (
        "Explain why an eval split should not be used for training.",
        "An eval split must stay held out so it can measure generalization.",
    ),
    (
        "What is reinforcement learning in one short sentence?",
        "Reinforcement learning improves a policy through rewards from interaction.",
    ),
    (
        "When should a model stop calling tools?",
        "It should stop after it has enough observations to answer the user.",
    ),
]


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_call(name: str, arguments: Dict) -> List[Dict]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def fmt_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def sft_tool_sample(category: str, prompt: str, tool_names: List[str], name: str, arguments: Dict, final_answer: str) -> Dict:
    result = execute_tool_call(name, arguments)
    return {
        "category": category,
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tool_names), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": json.dumps(tool_call(name, arguments), ensure_ascii=False),
            },
            {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer},
        ],
    }


def sft_no_tool_sample(prompt: str, final_answer: str) -> Dict:
    tools = ["calculate_math", "text_length", "unit_converter", "get_current_time"]
    return {
        "category": "stop_repair_no_tool",
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tools), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer},
        ],
    }


def math_expression(idx: int) -> Tuple[str, str]:
    a = 17 + idx
    b = 3 + (idx * 7) % 41
    kind = idx % 5
    if kind == 0:
        expression = f"{a}+{b}"
    elif kind == 1:
        expression = f"{a}*{b}"
    elif kind == 2:
        expression = f"{a + b}-{b}"
    elif kind == 3:
        root = 8 + idx
        expression = f"sqrt({root * root})"
    else:
        expression = f"{(idx % 23) + 2}**2"
    result = execute_tool_call("calculate_math", {"expression": expression})["result"]
    return expression, fmt_number(result)


def build_math_row(idx: int) -> Dict:
    expression, answer = math_expression(idx)
    if idx % 2 == 0:
        prompt = f"Use the calculator exactly once for case {idx}: {expression}, then give the final answer without another tool call."
    else:
        prompt = (
            f"\u8bf7\u7528\u8ba1\u7b97\u5668\u8ba1\u7b97 case {idx}: {expression}\uff0c"
            "\u62ff\u5230\u89c2\u6d4b\u540e\u7acb\u5373\u56de\u7b54\uff0c"
            "\u4e0d\u8981\u91cd\u590d\u8c03\u7528\u5de5\u5177\u3002"
        )
    return sft_tool_sample(
        "stop_repair_math",
        prompt,
        ["calculate_math", "text_length", "get_current_time"],
        "calculate_math",
        {"expression": expression},
        f"{expression} = {answer}.",
    )


def build_text_row(idx: int) -> Dict:
    text = f"{TEXTS[idx % len(TEXTS)]} case {idx}"
    field = "characters" if idx % 2 == 0 else "words"
    result = execute_tool_call("text_length", {"text": text})
    answer = fmt_number(result[field])
    if field == "characters":
        prompt = f"Count the characters in '{text}' once, then answer directly."
        final = f"The character count is {answer}."
    else:
        prompt = f"Count the words in '{text}' once, then answer directly."
        final = f"The word count is {answer}."
    return sft_tool_sample(
        "stop_repair_text",
        prompt,
        ["text_length", "calculate_math"],
        "text_length",
        {"text": text},
        final,
    )


def build_unit_row(idx: int) -> Dict:
    from_unit, to_unit = UNIT_PAIRS[idx % len(UNIT_PAIRS)]
    value = (idx % 97) + 1
    args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
    result = execute_tool_call("unit_converter", args)["result"]
    prompt = f"Convert case {idx}: {value} {from_unit} to {to_unit} with one tool call, then stop and answer."
    return sft_tool_sample(
        "stop_repair_unit",
        prompt,
        ["unit_converter", "calculate_math"],
        "unit_converter",
        args,
        f"{value} {from_unit} = {fmt_number(result)} {to_unit}.",
    )


def build_time_row(idx: int) -> Dict:
    timezone = "Asia/Shanghai" if idx % 2 == 0 else "America/New_York"
    args = {"timezone": timezone}
    result = execute_tool_call("get_current_time", args)
    prompt = f"Check the current time in {timezone} once for case {idx}, then answer with the returned datetime."
    return sft_tool_sample(
        "stop_repair_time",
        prompt,
        ["get_current_time", "calculate_math"],
        "get_current_time",
        args,
        f"The current time in {timezone} is {result['datetime']}.",
    )


def build_rows(total: int = DEFAULT_ROWS) -> List[Dict]:
    builders = [build_math_row, build_text_row, build_unit_row, build_time_row]
    rows: List[Dict] = []
    tool_rows = int(total * 0.85)
    for idx in range(tool_rows):
        rows.append(builders[idx % len(builders)](idx))
    for idx in range(total - tool_rows):
        prompt, answer = NO_TOOL_TOPICS[idx % len(NO_TOOL_TOPICS)]
        rows.append(sft_no_tool_sample(f"{prompt} Case {idx}.", answer))

    prompts = [row["conversations"][1]["content"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate prompts detected in stop-after-observation repair data")
    return rows


def summarize(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        category = row.get("category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT rows that teach stopping after a valid tool observation")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--output", default="dataset/tool_sft_stop_after_observation_repair.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.rows)
    print(f"stop repair rows: {len(rows)} {summarize(rows)}")
    if args.dry_run:
        print("dry run: no files written")
        return
    output = resolve_path(args.output)
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
