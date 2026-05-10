import json

from scripts.convert_minimind_agent_rl_data import (
    collect_converted_items,
    convert_row,
    convert_row_variants,
    extract_math_expressions,
    sft_sample_from_task,
    split_rows,
)


def source_row(prompt, gt, tools=None, final_content=""):
    tools = tools or ["calculate_math", "translate_text"]
    tool_schemas = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in tools]
    return {
        "conversations": [
            {"role": "system", "content": "", "tools": json.dumps(tool_schemas, ensure_ascii=False)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_content},
        ],
        "gt": gt,
    }


def test_extract_math_expressions_handles_parentheses_and_multiple_items():
    text = "帮我算一下(771-242)+84*27等于多少，另外算14**2，还有2935*998"

    assert extract_math_expressions(text) == ["(771-242)+84*27", "14**2", "2935*998"]


def test_convert_single_expression_to_current_math_schema():
    row = source_row("计算7109*2920", ["20758280"])

    converted = convert_row(row, "agent_rl_math", 1)

    assert converted["category"] == "converted_math_single"
    assert converted["source"] == "agent_rl_math"
    assert converted["tools"] == ["calculate_math", "text_length", "unit_converter", "get_current_time"]
    assert converted["expected_tool_sequence"] == ["calculate_math"]
    assert converted["expected_args"] == {"calculate_math": {"expression": "7109*2920"}}
    assert converted["expected_obs_keys"] == {"calculate_math": ["result"]}
    assert converted["expected_answer"] == "20758280"
    assert converted["gt"] == ["20758280"]
    assert converted["reward_profile"] == "dense_v2"


def test_convert_multi_expression_uses_repeated_tool_args():
    row = source_row("算(771-242)+84*27，14**2，2935*998", ["2797", "196", "2929130"])

    converted = convert_row(row, "agent_rl_math", 2)

    assert converted["category"] == "converted_math_multi"
    assert converted["expected_tool_sequence"] == ["calculate_math", "calculate_math", "calculate_math"]
    assert converted["expected_args"]["calculate_math"] == [
        {"expression": "(771-242)+84*27"},
        {"expression": "14**2"},
        {"expression": "2935*998"},
    ]
    assert converted["expected_answer"] == ["2797", "196", "2929130"]
    assert converted["max_tool_calls"] == 3


def test_convert_drops_unverified_or_nonempty_final_rows():
    assert convert_row(source_row("计算1+1", ["3"]), "agent_rl_math", 1) is None
    assert convert_row(source_row("计算1+1", ["2"], final_content="done"), "agent_rl_math", 2) is None
    assert convert_row(source_row("解释平方根", []), "agent_rl", 3) is None


def test_sft_sample_uses_current_tools_and_tool_observations():
    converted = convert_row(source_row("计算1+1和2*3", ["2", "6"]), "agent_rl_math", 1)

    sample = sft_sample_from_task(converted)
    conversations = sample["conversations"]

    assert conversations[0]["role"] == "system"
    tool_names = [tool["function"]["name"] for tool in json.loads(conversations[0]["tools"])]
    assert tool_names == ["calculate_math", "text_length", "unit_converter", "get_current_time"]
    assert conversations[1]["role"] == "user"
    assert [message["role"] for message in conversations if message["role"] == "tool"] == ["tool", "tool"]
    tool_calls = [message for message in conversations if message.get("tool_calls")]
    assert len(tool_calls) == 2
    assert json.loads(tool_calls[0]["tool_calls"])[0]["function"]["name"] == "calculate_math"
    assert conversations[-1]["content"] == "依次计算结果是 2、6。"


def test_split_rows_have_no_prompt_overlap():
    rows = [convert_row(source_row(f"计算{i}+1", [str(i + 1)]), "agent_rl_math", i) for i in range(10)]
    eval_rows, sft_rows, rl_rows = split_rows(rows, eval_size=3, sft_seed_size=4, seed=42)

    eval_prompts = {row["prompt"] for row in eval_rows}
    sft_prompts = {message["content"] for row in sft_rows for message in row["conversations"] if message["role"] == "user"}
    rl_prompts = {row["prompt"] for row in rl_rows}

    assert len(eval_rows) == 3
    assert len(sft_rows) == 4
    assert len(rl_rows) == 3
    assert not (eval_prompts & sft_prompts)
    assert not (eval_prompts & rl_prompts)
    assert not (sft_prompts & rl_prompts)


def test_collect_converted_items_deduplicates_prompts():
    rows = [
        source_row("计算1+1", ["2"]),
        source_row("计算1+1", ["2"]),
        source_row("解释一下", []),
    ]

    converted, stats = collect_converted_items([("agent_rl_math", rows)])

    assert len(converted) == 1
    assert stats["converted_rows"] == 1
    assert stats["duplicate_prompt"] == 1
    assert stats["empty_gt"] == 1


def test_normalized_prompt_mode_rewrites_single_expression_prompt():
    row = source_row("Please calculate 7109*2920", ["20758280"])

    converted = convert_row(row, "agent_rl_math", 1, prompt_mode="normalized")

    assert converted["prompt_mode"] == "normalized"
    assert converted["original_prompt"] == "Please calculate 7109*2920"
    assert converted["prompt"] == "Use the calculator tool to compute this expression: 7109*2920."
    assert converted["id"] == "converted_agent_rl_math_000001_normalized"
    assert converted["expected_args"] == {"calculate_math": {"expression": "7109*2920"}}


def test_normalized_prompt_mode_rewrites_multi_expression_prompt():
    row = source_row("Compute 1+1, 2*3, and 9-4", ["2", "6", "5"])

    converted = convert_row(row, "agent_rl_math", 2, prompt_mode="normalized")

    assert converted["category"] == "converted_math_multi"
    assert converted["prompt"] == (
        "Call the calculator tool once for each expression, in order, "
        "then give the results in the same order: 1+1; 2*3; 9-4."
    )
    assert converted["expected_tool_sequence"] == ["calculate_math", "calculate_math", "calculate_math"]


def test_both_prompt_mode_emits_original_and_normalized_variants():
    row = source_row("Please calculate 1+1", ["2"])

    variants = convert_row_variants(row, "agent_rl_math", 3, prompt_mode="both")

    assert [item["prompt_mode"] for item in variants] == ["original", "normalized"]
    assert variants[0]["prompt"] == "Please calculate 1+1"
    assert variants[1]["prompt"] == "Use the calculator tool to compute this expression: 1+1."


def test_convert_variants_can_filter_single_multi_and_expression_count():
    single = source_row("Compute 1+1", ["2"])
    multi = source_row("Compute 1+1, 2*3, and 9-4", ["2", "6", "5"])

    assert convert_row_variants(single, "agent_rl_math", 1, include_single=False) == []
    assert convert_row_variants(multi, "agent_rl_math", 2, include_multi=False) == []
    assert convert_row_variants(multi, "agent_rl_math", 3, max_expressions=2) == []


def test_collect_converted_items_can_emit_normalized_multi_only_rows():
    rows = [
        source_row("Compute 1+1", ["2"]),
        source_row("Compute 1+1 and 2*3", ["2", "6"]),
    ]

    converted, stats = collect_converted_items(
        [("agent_rl_math", rows)],
        prompt_mode="normalized",
        include_single=False,
        include_multi=True,
    )

    assert len(converted) == 1
    assert converted[0]["prompt_mode"] == "normalized"
    assert converted[0]["category"] == "converted_math_multi"
    assert stats["converted_rows"] == 1
