import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import execute_tool_call, get_tool_schemas


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_call(name, arguments):
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def sft_tool_sample(prompt, tool_names, name, arguments, final_answer):
    schemas = get_tool_schemas(tool_names)
    result = execute_tool_call(name, arguments)
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(schemas, ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(tool_call(name, arguments), ensure_ascii=False)},
            {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer},
        ]
    }


def sft_no_tool_sample(prompt, answer):
    return {"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def eval_item(
    task_id,
    category,
    prompt,
    tools,
    expected_sequence,
    expected_args=None,
    expected_answer=None,
    expected_obs_keys=None,
    args_match="equivalent",
    max_tool_calls=1,
    allow_no_tool=False,
):
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


def build_eval_rows():
    rows = []
    math_cases = [
        ("math_001", "帮我计算256*37", "256*37", "9472"),
        ("math_002", "请算一下 144 的平方根", "sqrt(144)", "12"),
        ("math_003", "What is 18 squared?", "18**2", "324"),
        ("math_004", "计算 2**10 + 24", "2**10+24", "1048"),
        ("math_005", "把 45 除以 9 再加 7", "45/9+7", "12"),
        ("math_006", "根号 625 是多少？", "sqrt(625)", "25"),
        ("math_007", "请计算 13 的平方", "13**2", "169"),
        ("math_008", "算一下 1000-376", "1000-376", "624"),
        ("math_009", "Compute sqrt(81)+9", "sqrt(81)+9", "18"),
        ("math_010", "3.5 乘以 8 等于多少？", "3.5*8", "28"),
    ]
    for task_id, prompt, expr, answer in math_cases:
        rows.append(
            eval_item(
                task_id,
                "single_tool_math",
                prompt,
                ["calculate_math", "get_current_time", "text_length"],
                ["calculate_math"],
                {"calculate_math": {"expression": expr}},
                answer,
                {"calculate_math": ["result"]},
            )
        )

    unit_cases = [
        ("unit_001", "帮我把100公里换算成英里", {"value": 100, "from_unit": "km", "to_unit": "miles"}, "62.137"),
        ("unit_002", "Convert 5 kg to pounds", {"value": 5, "from_unit": "kg", "to_unit": "pounds"}, "11.023"),
        ("unit_003", "30 摄氏度是多少华氏度？", {"value": 30, "from_unit": "celsius", "to_unit": "fahrenheit"}, "86"),
        ("unit_004", "10 miles 等于多少 km？", {"value": 10, "from_unit": "miles", "to_unit": "km"}, "16.093"),
        ("unit_005", "6 feet 换算成 meters", {"value": 6, "from_unit": "feet", "to_unit": "meters"}, "1.828"),
    ]
    for task_id, prompt, args, answer in unit_cases:
        rows.append(
            eval_item(
                task_id,
                "single_tool_unit",
                prompt,
                ["unit_converter", "calculate_math"],
                ["unit_converter"],
                {"unit_converter": args},
                answer,
                {"unit_converter": ["result"]},
            )
        )

    text_cases = [
        ("text_001", "统计 hello world 的字符数", {"text": "hello world"}, "11", "characters"),
        ("text_002", "How many characters are in MiniMind?", {"text": "MiniMind"}, "8", "characters"),
        ("text_003", "统计文本 Agentic RL 的长度", {"text": "Agentic RL"}, "10", "characters"),
        ("text_004", "请数一数 abc123 有几个字符", {"text": "abc123"}, "6", "characters"),
        ("text_005", "统计 OpenAI API 的单词数", {"text": "OpenAI API"}, "2", "words"),
    ]
    for task_id, prompt, args, answer, obs_key in text_cases:
        rows.append(
            eval_item(
                task_id,
                "single_tool_text",
                prompt,
                ["text_length", "calculate_math"],
                ["text_length"],
                {"text_length": args},
                answer,
                {"text_length": [obs_key]},
            )
        )

    time_cases = [
        ("time_001", "现在上海是什么时间？", {"timezone": "Asia/Shanghai"}),
        ("time_002", "What time is it in Asia/Shanghai?", {"timezone": "Asia/Shanghai"}),
        ("time_003", "请告诉我 America/New_York 当前时间", {"timezone": "America/New_York"}),
    ]
    for task_id, prompt, args in time_cases:
        rows.append(
            eval_item(
                task_id,
                "single_tool_time",
                prompt,
                ["get_current_time", "calculate_math"],
                ["get_current_time"],
                {"get_current_time": args},
                None,
            )
        )

    no_tool_cases = [
        ("notool_001", "解释什么是平方根"),
        ("notool_002", "用一句话说明什么是强化学习"),
        ("notool_003", "写一个简短的学习计划"),
        ("notool_004", "什么情况下应该使用工具调用？"),
        ("notool_005", "请礼貌地打个招呼"),
    ]
    for task_id, prompt in no_tool_cases:
        rows.append(
            eval_item(
                task_id,
                "no_tool",
                prompt,
                ["calculate_math", "text_length"],
                [],
                {},
                None,
                args_match="none",
                max_tool_calls=0,
                allow_no_tool=True,
            )
        )

    distractors = [
        ("dist_001", "请计算 19*21，不要查询时间", "19*21", "399"),
        ("dist_002", "虽然有文本工具，但我需要把9开方", "sqrt(9)", "3"),
        ("dist_003", "不要随机生成，直接计算 77+23", "77+23", "100"),
        ("dist_004", "请把 2 kg 换算成 pounds，不要做数学表达式", None, "4.409"),
        ("dist_005", "统计 abc def 的单词数，不要调用计算器", None, "2"),
    ]
    rows.extend(
        [
            eval_item("dist_001", "distractor", distractors[0][1], ["calculate_math", "get_current_time"], ["calculate_math"], {"calculate_math": {"expression": "19*21"}}, "399", {"calculate_math": ["result"]}),
            eval_item("dist_002", "distractor", distractors[1][1], ["calculate_math", "text_length"], ["calculate_math"], {"calculate_math": {"expression": "sqrt(9)"}}, "3", {"calculate_math": ["result"]}),
            eval_item("dist_003", "distractor", distractors[2][1], ["random_number", "calculate_math"], ["calculate_math"], {"calculate_math": {"expression": "77+23"}}, "100", {"calculate_math": ["result"]}),
            eval_item("dist_004", "distractor", distractors[3][1], ["unit_converter", "calculate_math"], ["unit_converter"], {"unit_converter": {"value": 2, "from_unit": "kg", "to_unit": "pounds"}}, "4.409", {"unit_converter": ["result"]}),
            eval_item("dist_005", "distractor", distractors[4][1], ["text_length", "calculate_math"], ["text_length"], {"text_length": {"text": "abc def"}}, "2", {"text_length": ["words"]}),
        ]
    )
    return rows


def build_sft_rows():
    rows = []
    for n in range(2, 96):
        square = n * n
        rows.append(sft_tool_sample(f"请计算 {n} 的平方", ["calculate_math", "text_length"], "calculate_math", {"expression": f"{n}**2"}, f"{n} 的平方是 {square}。"))
        rows.append(sft_tool_sample(f"请计算 {square} 的平方根", ["calculate_math"], "calculate_math", {"expression": f"sqrt({square})"}, f"{square} 的平方根是 {n}。"))
        rows.append(sft_tool_sample(f"根号 {square} 是多少？", ["calculate_math", "text_length"], "calculate_math", {"expression": f"sqrt({square})"}, f"根号 {square} 是 {n}。"))
        rows.append(sft_tool_sample(f"把 {square} 开方", ["calculate_math", "text_length"], "calculate_math", {"expression": f"sqrt({square})"}, f"{square} 开方等于 {n}。"))

    targeted_math = [
        ("根号 625 是多少？", "sqrt(625)", "根号 625 是 25。"),
        ("625 的平方根是多少？", "sqrt(625)", "625 的平方根是 25。"),
        ("把 625 开方", "sqrt(625)", "625 开方等于 25。"),
        ("What is the square root of 625?", "sqrt(625)", "The square root of 625 is 25."),
        ("请计算 13 的平方", "13**2", "13 的平方是 169。"),
        ("What is 18 squared?", "18**2", "18 squared is 324."),
        ("算一下 1000-376", "1000-376", "1000-376 = 624。"),
        ("1000 减 376 等于多少？", "1000-376", "1000 减 376 等于 624。"),
        ("把 45 除以 9 再加 7", "45/9+7", "45 除以 9 再加 7 等于 12。"),
        ("3.5 乘以 8 等于多少？", "3.5*8", "3.5 乘以 8 等于 28。"),
    ]
    for prompt, expression, final_answer in targeted_math:
        rows.append(sft_tool_sample(prompt, ["calculate_math", "text_length"], "calculate_math", {"expression": expression}, final_answer))

    for value in [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]:
        result = execute_tool_call("unit_converter", {"value": value, "from_unit": "km", "to_unit": "miles"})["result"]
        rows.append(sft_tool_sample(f"帮我把 {value} 公里换算成英里", ["unit_converter", "calculate_math"], "unit_converter", {"value": value, "from_unit": "km", "to_unit": "miles"}, f"{value} km = {result} miles。"))

    targeted_units = [
        ("Convert 5 kg to pounds", {"value": 5, "from_unit": "kg", "to_unit": "pounds"}, "5 kg = {result} pounds."),
        ("30 摄氏度是多少华氏度？", {"value": 30, "from_unit": "celsius", "to_unit": "fahrenheit"}, "30 摄氏度 = {result} 华氏度。"),
        ("6 feet 换算成 meters", {"value": 6, "from_unit": "feet", "to_unit": "meters"}, "6 feet = {result} meters."),
        ("请把 2 kg 换算成 pounds，不要做数学表达式", {"value": 2, "from_unit": "kg", "to_unit": "pounds"}, "2 kg = {result} pounds."),
    ]
    for prompt, args, template in targeted_units:
        result = execute_tool_call("unit_converter", args)["result"]
        rows.append(sft_tool_sample(prompt, ["unit_converter", "calculate_math"], "unit_converter", args, template.format(result=result)))

    for text in ["hello world", "MiniMind", "Agentic RL", "OpenAI API", "abc123", "tool use"]:
        result = execute_tool_call("text_length", {"text": text})
        rows.append(sft_tool_sample(f"统计 {text} 的字符数", ["text_length", "calculate_math"], "text_length", {"text": text}, f"这段文本共有 {result['characters']} 个字符。"))

    targeted_texts = [
        ("统计 hello world 的字符数", {"text": "hello world"}, "这段文本共有 {characters} 个字符。"),
        ("统计文本 Agentic RL 的长度", {"text": "Agentic RL"}, "这段文本共有 {characters} 个字符。"),
        ("统计 OpenAI API 的单词数", {"text": "OpenAI API"}, "这段文本共有 {words} 个单词。"),
        ("请数一数 abc123 有几个字符", {"text": "abc123"}, "这段文本共有 {characters} 个字符。"),
        ("统计 abc def 的单词数，不要调用计算器", {"text": "abc def"}, "这段文本共有 {words} 个单词。"),
    ]
    for prompt, args, template in targeted_texts:
        result = execute_tool_call("text_length", args)
        rows.append(sft_tool_sample(prompt, ["text_length", "calculate_math"], "text_length", args, template.format(**result)))

    distractor_sft = [
        ("请计算 19*21，不要查询时间", ["calculate_math", "get_current_time"], "calculate_math", {"expression": "19*21"}, "19*21 = 399。"),
        ("虽然有文本工具，但我需要把9开方", ["calculate_math", "text_length"], "calculate_math", {"expression": "sqrt(9)"}, "9 开方等于 3。"),
        ("不要随机生成，直接计算 77+23", ["random_number", "calculate_math"], "calculate_math", {"expression": "77+23"}, "77+23 = 100。"),
    ]
    for prompt, tool_names, name, args, final_answer in distractor_sft:
        rows.append(sft_tool_sample(prompt, tool_names, name, args, final_answer))

    rows.extend(
        [
            sft_no_tool_sample("解释什么是平方根", "平方根是指一个数乘以自己后得到原数的值。"),
            sft_no_tool_sample("用一句话说明什么是强化学习", "强化学习是智能体通过行动、反馈和奖励来学习策略的方法。"),
            sft_no_tool_sample("请礼貌地打个招呼", "你好，很高兴见到你。"),
        ]
    )
    return rows


def build_rl_rows(eval_rows):
    rows = []
    for item in eval_rows:
        if item["category"] in {"single_tool_math", "single_tool_unit", "single_tool_text", "single_tool_time", "no_tool"}:
            row = dict(item)
            row["gt"] = [] if item.get("expected_answer") is None else [str(item["expected_answer"])]
            rows.append(row)
    return rows


def main():
    eval_rows = build_eval_rows()
    sft_rows = build_sft_rows()
    rl_rows = build_rl_rows(eval_rows)

    write_jsonl(PROJECT_ROOT / "evals" / "tool_eval.jsonl", eval_rows)
    write_jsonl(PROJECT_ROOT / "dataset" / "tool_sft_small.jsonl", sft_rows)
    write_jsonl(PROJECT_ROOT / "dataset" / "agent_rl_tooluse.jsonl", rl_rows)
    (PROJECT_ROOT / "evals" / "reports").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "evals" / "reports" / ".gitkeep").write_text("", encoding="utf-8")

    print(f"eval rows: {len(eval_rows)} -> evals/tool_eval.jsonl")
    print(f"sft rows: {len(sft_rows)} -> dataset/tool_sft_small.jsonl")
    print(f"rl rows: {len(rl_rows)} -> dataset/agent_rl_tooluse.jsonl")


if __name__ == "__main__":
    main()
