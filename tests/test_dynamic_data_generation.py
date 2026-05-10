import json

from scripts.make_agentic_tooluse_dynamic_data import (
    DYNAMIC_CATEGORIES,
    build_all_splits,
    default_output_paths,
    parse_levels,
    row_prompt,
    summarize,
)


def test_dynamic_data_generation_builds_level_splits_without_overlap():
    splits = build_all_splits(eval_size=24, train_size=48, sft_seed_size=48, language="zh")

    assert len(splits["l1"]) == 24
    assert len(splits["l2"]) == 24
    assert len(splits["l3"]) == 24
    assert len(splits["l4"]) == 24
    assert len(splits["train"]) == 48
    assert len(splits["sft_seed"]) == 48

    prompt_sets = {name: {row_prompt(row) for row in rows} for name, rows in splits.items()}
    names = list(prompt_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not (prompt_sets[left] & prompt_sets[right])

    for key in ["l1", "l2", "l3", "l4", "train"]:
        assert set(summarize(splits[key])) == set(DYNAMIC_CATEGORIES)
        assert all(count > 0 for count in summarize(splits[key]).values())
        assert {row["language"] for row in splits[key]} == {"zh"}
    assert {row["language"] for row in splits["sft_seed"]} == {"zh"}


def test_dynamic_rl_rows_include_dense_reward_fields_and_three_step_cases():
    rows = build_all_splits(eval_size=24, train_size=48, language="zh")["train"]

    assert all(row["reward_profile"] == "dense_v2" for row in rows)
    assert all("difficulty_level" in row for row in rows)
    assert all("language" in row for row in rows)
    assert all("gt" in row for row in rows)
    assert all("expected_answer" in row for row in rows)

    three_step_rows = [row for row in rows if len(row["expected_tool_sequence"]) == 3]
    assert three_step_rows
    assert all(row["max_tool_calls"] == 3 for row in three_step_rows)

    repeated_tool_rows = [
        row
        for row in rows
        if row["expected_tool_sequence"].count("calculate_math") >= 2
    ]
    assert repeated_tool_rows
    assert isinstance(repeated_tool_rows[0]["expected_args"]["calculate_math"], list)


def test_dynamic_sft_seed_rows_are_tool_conversations():
    rows = build_all_splits(eval_size=24, train_size=48, sft_seed_size=48, language="zh")["sft_seed"]

    tool_rows = [row for row in rows if any(message.get("role") == "tool" for message in row["conversations"])]
    no_tool_rows = [row for row in rows if not any(message.get("role") == "tool" for message in row["conversations"])]

    assert tool_rows
    assert no_tool_rows

    sample = tool_rows[0]
    conversations = sample["conversations"]
    assert conversations[0]["role"] == "system"
    assert json.loads(conversations[0]["tools"])
    assert conversations[1]["role"] == "user"
    assert conversations[-1]["role"] == "assistant"
    assert "最终答案" in conversations[-1]["content"]

    assistant_tool_turns = [message for message in conversations if message.get("tool_calls")]
    assert assistant_tool_turns
    assert json.loads(assistant_tool_turns[0]["tool_calls"])[0]["type"] == "function"
    assert any(message.get("role") == "tool" for message in conversations)

    no_tool_conversations = no_tool_rows[0]["conversations"]
    assert no_tool_conversations[0]["role"] == "system"
    assert no_tool_conversations[-1]["role"] == "assistant"
    assert not any(message.get("tool_calls") for message in no_tool_conversations)


def test_dynamic_generation_accepts_configurable_levels():
    splits = build_all_splits(
        eval_size=24,
        train_size=48,
        sft_seed_size=24,
        train_levels=[1, 2],
        sft_seed_levels=[2],
        language="zh",
    )

    assert {row["difficulty_level"] for row in splits["train"]} == {1, 2}
    assert parse_levels("1, 3,4") == [1, 3, 4]
    assert len(splits["sft_seed"]) == 24


def test_dynamic_generation_supports_zh_and_mixed_languages():
    zh_splits = build_all_splits(eval_size=24, train_size=48, sft_seed_size=24, language="zh")
    zh_prompt = row_prompt(zh_splits["l1"][0])

    assert "统计" in zh_prompt
    assert "忽略干扰数字" in zh_prompt
    assert {row["language"] for row in zh_splits["train"]} == {"zh"}
    assert {row["language"] for row in zh_splits["sft_seed"]} == {"zh"}

    mixed_splits = build_all_splits(eval_size=24, train_size=48, sft_seed_size=24, language="mixed")
    mixed_languages = {row["language"] for row in mixed_splits["train"]}
    assert mixed_languages == {"zh", "en"}

    prompt_sets = {name: {row_prompt(row) for row in rows} for name, rows in mixed_splits.items()}
    names = list(prompt_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not (prompt_sets[left] & prompt_sets[right])


def test_dynamic_default_output_paths_are_language_aware():
    assert default_output_paths("en")["l1_path"] == "evals/tool_eval_dynamic_l1.jsonl"
    assert default_output_paths("en")["train_path"] == "dataset/agent_rl_tooluse_dynamic_train.jsonl"
    assert default_output_paths("en")["sft_seed_path"] == "dataset/tool_sft_dynamic_seed.jsonl"

    assert default_output_paths("zh")["l1_path"] == "evals/tool_eval_dynamic_zh_l1.jsonl"
    assert default_output_paths("zh")["train_path"] == "dataset/agent_rl_tooluse_dynamic_zh_train.jsonl"
    assert default_output_paths("zh")["sft_seed_path"] == "dataset/tool_sft_dynamic_zh_seed.jsonl"

    assert default_output_paths("mixed")["l1_path"] == "evals/tool_eval_dynamic_mixed_l1.jsonl"
    assert default_output_paths("mixed")["train_path"] == "dataset/agent_rl_tooluse_dynamic_mixed_train.jsonl"
    assert default_output_paths("mixed")["sft_seed_path"] == "dataset/tool_sft_dynamic_mixed_seed.jsonl"
