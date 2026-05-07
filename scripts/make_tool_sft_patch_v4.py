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


def no_tool_with_tools_sample(prompt, tool_names, final_answer):
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tool_names), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_answer},
        ]
    }


def tool_response_final_sample(prompt, tool_names, name, arguments, final_answer):
    result = execute_tool_call(name, arguments)
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(tool_names), ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(tool_call(name, arguments), ensure_ascii=False)},
            {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer.format(**result)},
        ]
    }


def build_rows():
    rows = []
    no_tool_tools = ["calculate_math", "text_length"]
    rows.extend(
        [
            no_tool_with_tools_sample(
                "解释什么是平方根",
                no_tool_tools,
                "平方根是指一个数与自身相乘后等于原数的值。例如，3 是 9 的平方根，因为 3×3=9。",
            ),
            no_tool_with_tools_sample(
                "什么情况下应该使用工具调用？",
                no_tool_tools,
                "当问题需要精确计算、单位换算、查询当前时间或统计文本等外部功能时，应该使用工具调用；普通解释、问候和写作类问题通常不需要工具。",
            ),
            no_tool_with_tools_sample(
                "请礼貌地打个招呼",
                no_tool_tools,
                "你好，很高兴见到你！希望你今天一切顺利。",
            ),
            no_tool_with_tools_sample(
                "用一句话说明什么是强化学习",
                no_tool_tools,
                "强化学习是智能体通过与环境交互并根据奖励反馈不断改进决策策略的方法。",
            ),
            no_tool_with_tools_sample(
                "写一个简短的学习计划",
                no_tool_tools,
                "可以先确定学习目标，再把内容拆成每日小任务，每天学习、练习并复盘，最后用小项目检验掌握情况。",
            ),
        ]
    )

    rows.extend(
        [
            tool_response_final_sample(
                "算一下 1000-376",
                ["calculate_math", "get_current_time", "text_length"],
                "calculate_math",
                {"expression": "1000-376"},
                "1000-376 = {result}。",
            ),
            tool_response_final_sample(
                "Compute sqrt(81)+9",
                ["calculate_math", "get_current_time", "text_length"],
                "calculate_math",
                {"expression": "sqrt(81)+9"},
                "sqrt(81)+9 = {result}。",
            ),
            tool_response_final_sample(
                "10 miles 等于多少 km？",
                ["unit_converter", "calculate_math"],
                "unit_converter",
                {"value": 10, "from_unit": "miles", "to_unit": "km"},
                "10 miles = {result} km。",
            ),
            tool_response_final_sample(
                "统计 OpenAI API 的单词数",
                ["text_length", "calculate_math"],
                "text_length",
                {"text": "OpenAI API"},
                "OpenAI API 有 {words} 个单词。",
            ),
            tool_response_final_sample(
                "虽然有文本工具，但我需要把9开方",
                ["calculate_math", "text_length"],
                "calculate_math",
                {"expression": "sqrt(9)"},
                "sqrt(9) = {result}。",
            ),
            tool_response_final_sample(
                "What time is it in Asia/Shanghai?",
                ["get_current_time", "calculate_math"],
                "get_current_time",
                {"timezone": "Asia/Shanghai"},
                "The current time in Asia/Shanghai is {datetime}.",
            ),
        ]
    )
    return rows


def main():
    rows = build_rows()
    output = PROJECT_ROOT / "dataset" / "tool_sft_patch_v4.jsonl"
    write_jsonl(output, rows)
    print(f"patch rows: {len(rows)} -> dataset/tool_sft_patch_v4.jsonl")


if __name__ == "__main__":
    main()
