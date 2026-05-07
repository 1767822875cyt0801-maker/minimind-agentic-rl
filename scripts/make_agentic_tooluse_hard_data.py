import json
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


def fmt_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


def sft_no_tool_with_tools(prompt, tools, final_answer):
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tools), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer},
        ]
    }


def sft_multi_tool_sample(prompt, tools, steps, final_answer):
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


def add_text_math_cases(eval_rows, sft_rows):
    cases = [
        ("hard_tm_001", "统计 OpenAI API 的字符数，然后把字符数乘以3", "OpenAI API", "characters", 3, "10*3", "30"),
        ("hard_tm_002", "统计 abc def gh 的单词数，然后加上7", "abc def gh", "words", 7, "3+7", "10"),
        ("hard_tm_003", "How many words are in Agentic RL, then multiply that by 9?", "Agentic RL", "words", 9, "2*9", "18"),
        ("hard_tm_004", "数一数 MiniMind Agent 的字符数，再减去4", "MiniMind Agent", "characters", -4, "14-4", "10"),
        ("hard_tm_005", "统计 tool use eval 的单词数，然后计算它的平方", "tool use eval", "words", None, "3**2", "9"),
        ("hard_tm_006", "先统计 abc123 的字符数，再计算字符数除以2", "abc123", "characters", None, "6/2", "3"),
    ]
    for task_id, prompt, text, field, delta, expression, answer in cases:
        steps = [("text_length", {"text": text}), ("calculate_math", {"expression": expression})]
        eval_rows.append(
            eval_item(
                task_id,
                "hard_multistep_text_math",
                prompt,
                ["text_length", "calculate_math", "unit_converter"],
                ["text_length", "calculate_math"],
                {"text_length": {"text": text}, "calculate_math": {"expression": expression}},
                answer,
                {"text_length": [field], "calculate_math": ["result"]},
                max_tool_calls=2,
            )
        )
        sft_rows.append(
            sft_multi_tool_sample(
                prompt,
                ["text_length", "calculate_math", "unit_converter"],
                steps,
                f"计算结果是 {answer}。",
            )
        )


def add_math_unit_cases(eval_rows, sft_rows):
    cases = [
        ("hard_mu_001", "先算 12*5，再把这个数值的 km 换算成 miles", "12*5", 60, "km", "miles"),
        ("hard_mu_002", "先计算 3+2，然后把结果 kg 换算成 pounds", "3+2", 5, "kg", "pounds"),
        ("hard_mu_003", "先算 100-70，再把结果摄氏度换算成华氏度", "100-70", 30, "celsius", "fahrenheit"),
        ("hard_mu_004", "先计算 18/3，然后把结果 feet 换算成 meters", "18/3", 6, "feet", "meters"),
        ("hard_mu_005", "Calculate 7*8, then convert that many miles to km", "7*8", 56, "miles", "km"),
        ("hard_mu_006", "先算 sqrt(81)，再把这个数值的 kg 换算成 pounds", "sqrt(81)", 9, "kg", "pounds"),
    ]
    for task_id, prompt, expression, value, from_unit, to_unit in cases:
        unit_args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
        unit_result = execute_tool_call("unit_converter", unit_args)["result"]
        answer = str(unit_result)
        steps = [("calculate_math", {"expression": expression}), ("unit_converter", unit_args)]
        eval_rows.append(
            eval_item(
                task_id,
                "hard_multistep_math_unit",
                prompt,
                ["calculate_math", "unit_converter", "text_length"],
                ["calculate_math", "unit_converter"],
                {"calculate_math": {"expression": expression}, "unit_converter": unit_args},
                answer,
                {"calculate_math": ["result"], "unit_converter": ["result"]},
                max_tool_calls=2,
            )
        )
        sft_rows.append(
            sft_multi_tool_sample(
                prompt,
                ["calculate_math", "unit_converter", "text_length"],
                steps,
                f"{fmt_number(value)} {from_unit} = {answer} {to_unit}。",
            )
        )


def add_field_selection_cases(eval_rows, sft_rows):
    text_cases = [
        ("hard_field_001", "统计 OpenAI API 的单词数，不要回答字符数", "OpenAI API", "words", "2", "OpenAI API 有 2 个单词。"),
        ("hard_field_002", "统计 OpenAI API 的字符数，不要回答单词数", "OpenAI API", "characters", "10", "OpenAI API 有 10 个字符。"),
        ("hard_field_003", "How many words are in hello world?", "hello world", "words", "2", "hello world has 2 words."),
        ("hard_field_004", "How many characters are in hello world?", "hello world", "characters", "11", "hello world has 11 characters."),
    ]
    for task_id, prompt, text, field, answer, final_answer in text_cases:
        args = {"text": text}
        eval_rows.append(
            eval_item(
                task_id,
                "hard_field_selection",
                prompt,
                ["text_length", "calculate_math"],
                ["text_length"],
                {"text_length": args},
                answer,
                {"text_length": [field]},
            )
        )
        sft_rows.append(sft_multi_tool_sample(prompt, ["text_length", "calculate_math"], [("text_length", args)], final_answer))

    unit_cases = [
        ("hard_field_005", "Convert 5 kg to pounds; answer the converted result, not the input value", {"value": 5, "from_unit": "kg", "to_unit": "pounds"}, "5 kg = {result} pounds."),
        ("hard_field_006", "30 摄氏度是多少华氏度？只回答换算后的结果", {"value": 30, "from_unit": "celsius", "to_unit": "fahrenheit"}, "30 摄氏度 = {result} 华氏度。"),
        ("hard_field_007", "6 feet 换算成 meters，只用换算结果作答", {"value": 6, "from_unit": "feet", "to_unit": "meters"}, "6 feet = {result} meters."),
    ]
    for task_id, prompt, args, template in unit_cases:
        result = execute_tool_call("unit_converter", args)
        answer = str(result["result"])
        eval_rows.append(
            eval_item(
                task_id,
                "hard_field_selection",
                prompt,
                ["unit_converter", "calculate_math"],
                ["unit_converter"],
                {"unit_converter": args},
                answer,
                {"unit_converter": ["result"]},
            )
        )
        sft_rows.append(
            sft_multi_tool_sample(
                prompt,
                ["unit_converter", "calculate_math"],
                [("unit_converter", args)],
                template.format(**result),
            )
        )


def add_distractor_cases(eval_rows, sft_rows):
    cases = [
        ("hard_dist_001", "不要查询时间，直接计算 123+456", ["calculate_math", "get_current_time"], "calculate_math", {"expression": "123+456"}, "579"),
        ("hard_dist_002", "不要调用计算器，统计 red blue green 的单词数", ["text_length", "calculate_math"], "text_length", {"text": "red blue green"}, "3"),
        ("hard_dist_003", "不要随机生成，直接计算 88+12", ["random_number", "calculate_math"], "calculate_math", {"expression": "88+12"}, "100"),
        ("hard_dist_004", "虽然有文本工具，但请计算 sqrt(144)+6", ["calculate_math", "text_length"], "calculate_math", {"expression": "sqrt(144)+6"}, "18"),
        ("hard_dist_005", "虽然有计算器，但请把 3 kg 换算成 pounds", ["unit_converter", "calculate_math"], "unit_converter", {"value": 3, "from_unit": "kg", "to_unit": "pounds"}, "6.614"),
        ("hard_dist_006", "不要查时间也不要随机生成，计算 19*21", ["get_current_time", "random_number", "calculate_math"], "calculate_math", {"expression": "19*21"}, "399"),
    ]
    for task_id, prompt, tools, name, args, answer in cases:
        expected_obs = {name: ["words"]} if name == "text_length" else {name: ["result"]}
        eval_rows.append(
            eval_item(
                task_id,
                "hard_distractor",
                prompt,
                tools,
                [name],
                {name: args},
                answer,
                expected_obs,
            )
        )
        if name == "text_length":
            final_answer = f"这段文本共有 {answer} 个单词。"
        elif name == "unit_converter":
            result = execute_tool_call(name, args)
            final_answer = f"{fmt_number(args['value'])} {args['from_unit']} = {result['result']} {args['to_unit']}。"
        else:
            final_answer = f"{args['expression']} = {answer}。"
        sft_rows.append(sft_multi_tool_sample(prompt, tools, [(name, args)], final_answer))


def add_no_tool_cases(eval_rows, sft_rows):
    cases = [
        ("hard_notool_001", "解释为什么平方根不是平方", "平方根是求能自乘得到原数的值，平方是把一个数自乘。"),
        ("hard_notool_002", "请给我一句鼓励学习的话", "保持稳定的小步前进，今天的认真会变成明天的底气。"),
        ("hard_notool_003", "什么情况下不应该使用工具调用？", "当问题只需要解释、建议、写作或普通对话时，通常不需要使用工具调用。"),
        ("hard_notool_004", "简述 Agentic RL 的目标", "Agentic RL 的目标是让模型通过奖励信号学会更可靠地规划、调用工具并使用观察结果完成任务。"),
        ("hard_notool_005", "写一个两步学习计划", "第一步明确目标并学习核心概念；第二步通过练习和复盘巩固。"),
        ("hard_notool_006", "礼貌地向老师问好", "老师您好，感谢您的指导。"),
    ]
    tools = ["calculate_math", "text_length", "get_current_time"]
    for task_id, prompt, answer in cases:
        eval_rows.append(
            eval_item(
                task_id,
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
        )
        sft_rows.append(sft_no_tool_with_tools(prompt, tools, answer))


def build_eval_and_sft_rows():
    eval_rows = []
    sft_rows = []
    add_text_math_cases(eval_rows, sft_rows)
    add_math_unit_cases(eval_rows, sft_rows)
    add_field_selection_cases(eval_rows, sft_rows)
    add_distractor_cases(eval_rows, sft_rows)
    add_no_tool_cases(eval_rows, sft_rows)
    return eval_rows, sft_rows


def build_rl_rows(eval_rows):
    rows = []
    for item in eval_rows:
        row = dict(item)
        row["gt"] = [] if item.get("expected_answer") is None else [str(item["expected_answer"])]
        rows.append(row)
    return rows


def main():
    eval_rows, sft_rows = build_eval_and_sft_rows()
    rl_rows = build_rl_rows(eval_rows)

    write_jsonl(PROJECT_ROOT / "evals" / "tool_eval_hard.jsonl", eval_rows)
    write_jsonl(PROJECT_ROOT / "dataset" / "agent_rl_tooluse_hard.jsonl", rl_rows)
    write_jsonl(PROJECT_ROOT / "dataset" / "tool_sft_hard_small.jsonl", sft_rows)
    (PROJECT_ROOT / "evals" / "reports").mkdir(parents=True, exist_ok=True)

    print(f"hard eval rows: {len(eval_rows)} -> evals/tool_eval_hard.jsonl")
    print(f"hard rl rows: {len(rl_rows)} -> dataset/agent_rl_tooluse_hard.jsonl")
    print(f"hard sft rows: {len(sft_rows)} -> dataset/tool_sft_hard_small.jsonl")


if __name__ == "__main__":
    main()
