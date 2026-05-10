from agent.reward import answer_contains_expected, compute_total_reward
from agent.tool_env import ToolUseEnv
from agent.trajectory import ToolTrajectory, TrajectoryStep


def build_math_traj(final_answer):
    env = ToolUseEnv(mode="raw")
    env.reset({"id": "math_001", "prompt": "算 256*37", "tools": ["calculate_math"]})
    env.step('<tool_call>{"name":"calculate_math","arguments":{"expression":"256*37"}}</tool_call>')
    env.step(final_answer)
    return env.get_trajectory()


def build_single_tool_traj(tool_name, arguments, result, final_answer):
    return ToolTrajectory(
        task_id="test_case",
        prompt="test prompt",
        tools=[tool_name],
        steps=[
            TrajectoryStep(role="user", type="user", content="test prompt"),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": tool_name, "arguments": arguments},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": tool_name, "arguments": arguments, "result": result},
            ),
            TrajectoryStep(role="assistant", type="assistant_final", content=final_answer),
        ],
        final_answer=final_answer,
        done=True,
    )


def build_final_traj(final_answer):
    return ToolTrajectory(
        task_id="test_case",
        prompt="test prompt",
        tools=["calculate_math", "text_length"],
        steps=[
            TrajectoryStep(role="user", type="user", content="test prompt"),
            TrajectoryStep(role="assistant", type="assistant_final", content=final_answer),
        ],
        final_answer=final_answer,
        done=True,
    )


def test_reward_prefers_correct_final_answer():
    task = {
        "id": "math_001",
        "tools": ["calculate_math"],
        "expected_tool_sequence": ["calculate_math"],
        "expected_args": {"calculate_math": {"expression": "256*37"}},
        "expected_answer": "9472",
        "max_tool_calls": 1,
    }
    good_reward, good_info = compute_total_reward(build_math_traj("256*37 = 9472"), task)
    bad_reward, bad_info = compute_total_reward(build_math_traj("256*37 = 1111"), task)
    assert good_info["answer_acc"]
    assert not bad_info["answer_acc"]
    assert good_reward > bad_reward


def test_number_answer_does_not_match_substrings():
    assert not answer_contains_expected(
        "根号 625 是 390625。",
        "25",
        expected_answer_type="number",
    )


def test_number_answer_allows_reasonable_rounding():
    assert answer_contains_expected(
        "100公里等于约62.14英里。",
        "62.137",
        expected_answer_type="number",
    )


def test_number_answer_extracts_mixed_language_numbers():
    assert answer_contains_expected("18 squared is 324.", "324", expected_answer_type="number")
    assert answer_contains_expected("13的平方是169。", "169", expected_answer_type="number")
    assert answer_contains_expected("数一数有6个字符，1个单词。", "6", expected_answer_type="number")


def test_typed_number_reward_rejects_substring_false_positive():
    reward, info = compute_total_reward(
        build_math_traj("根号 625 是 390625。"),
        {
            "id": "math_006",
            "tools": ["calculate_math"],
            "expected_tool_sequence": ["calculate_math"],
            "expected_args": {"calculate_math": {"expression": "sqrt(625)"}},
            "expected_answer": "25",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert not info["answer_acc"]
    assert info["final_answer_reward"] == -0.8
    assert reward < 2.0


def test_obs_use_matches_chinese_datetime_format():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "get_current_time",
            {"timezone": "Asia/Shanghai"},
            {"datetime": "2026-05-05 20:49:27", "timezone": "Asia/Shanghai"},
            "现在上海的时间是2026年5月5日20时49分27秒。",
        ),
        {
            "expected_tool_sequence": ["get_current_time"],
            "expected_args": {"get_current_time": {"timezone": "Asia/Shanghai"}},
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]


def test_obs_use_matches_english_datetime_format():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "get_current_time",
            {"timezone": "Asia/Shanghai"},
            {"datetime": "2026-05-05 20:49:27", "timezone": "Asia/Shanghai"},
            "The current time in Asia/Shanghai is 20:49:27 on May 5, 2026.",
        ),
        {
            "expected_tool_sequence": ["get_current_time"],
            "expected_args": {"get_current_time": {"timezone": "Asia/Shanghai"}},
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]


def test_time_wrong_timezone_counts_as_wrong_args():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "get_current_time",
            {"timezone": "Asia/Shanghai"},
            {"datetime": "2026-05-05 20:49:27", "timezone": "Asia/Shanghai"},
            "当前时间是2026年5月5日20时49分27秒。",
        ),
        {
            "expected_tool_sequence": ["get_current_time"],
            "expected_args": {"get_current_time": {"timezone": "America/New_York"}},
            "max_tool_calls": 1,
        },
    )
    assert not info["args_acc"]
    assert "wrong_args" in info["failures"]


def test_obs_use_matches_numeric_tool_results_with_rounding():
    _, unit_info = compute_total_reward(
        build_single_tool_traj(
            "unit_converter",
            {"value": 5, "from_unit": "kg", "to_unit": "pounds"},
            {"value": 5.0, "from_unit": "kg", "to_unit": "pounds", "result": 11.023113},
            "5 kg is approximately 11.02 pounds.",
        ),
        {
            "expected_tool_sequence": ["unit_converter"],
            "expected_args": {"unit_converter": {"value": 5, "from_unit": "kg", "to_unit": "pounds"}},
            "max_tool_calls": 1,
        },
    )
    _, text_info = compute_total_reward(
        build_single_tool_traj(
            "text_length",
            {"text": "OpenAI API"},
            {"text": "OpenAI API", "characters": 10, "words": 2},
            "这段文本共有 10 个字符，2 个单词。",
        ),
        {
            "expected_tool_sequence": ["text_length"],
            "expected_args": {"text_length": {"text": "OpenAI API"}},
            "max_tool_calls": 1,
        },
    )
    assert unit_info["obs_use_acc"]
    assert text_info["obs_use_acc"]


def test_expected_obs_keys_rejects_wrong_text_length_field():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "text_length",
            {"text": "OpenAI API"},
            {"text": "OpenAI API", "characters": 10, "words": 2},
            "这段文本共有 10 个单词。",
        ),
        {
            "expected_tool_sequence": ["text_length"],
            "expected_args": {"text_length": {"text": "OpenAI API"}},
            "expected_obs_keys": {"text_length": ["words"]},
            "expected_answer": "2",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert not info["obs_use_acc"]
    assert "observation_ignored" in info["failures"]
    assert not info["answer_acc"]


def test_expected_obs_keys_accepts_requested_text_length_field():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "text_length",
            {"text": "OpenAI API"},
            {"text": "OpenAI API", "characters": 10, "words": 2},
            "这段文本共有 2 个单词。",
        ),
        {
            "expected_tool_sequence": ["text_length"],
            "expected_args": {"text_length": {"text": "OpenAI API"}},
            "expected_obs_keys": {"text_length": ["words"]},
            "expected_answer": "2",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]
    assert info["answer_acc"]


def test_expected_obs_keys_missing_tool_falls_back_to_default_keys():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "unit_converter",
            {"value": 5, "from_unit": "kg", "to_unit": "pounds"},
            {"value": 5.0, "from_unit": "kg", "to_unit": "pounds", "result": 11.023113},
            "5 kg 约等于 11.02 pounds。",
        ),
        {
            "expected_tool_sequence": ["unit_converter"],
            "expected_args": {"unit_converter": {"value": 5, "from_unit": "kg", "to_unit": "pounds"}},
            "expected_obs_keys": {"text_length": ["words"]},
            "expected_answer": "11.023",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]
    assert info["answer_acc"]


def test_expected_obs_keys_accepts_string_value():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "text_length",
            {"text": "OpenAI API"},
            {"text": "OpenAI API", "characters": 10, "words": 2},
            "这段文本共有 2 个单词。",
        ),
        {
            "expected_tool_sequence": ["text_length"],
            "expected_args": {"text_length": {"text": "OpenAI API"}},
            "expected_obs_keys": {"text_length": "words"},
            "expected_answer": "2",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]
    assert info["answer_acc"]


def test_expected_obs_keys_empty_list_falls_back_to_default_keys():
    _, info = compute_total_reward(
        build_single_tool_traj(
            "text_length",
            {"text": "OpenAI API"},
            {"text": "OpenAI API", "characters": 10, "words": 2},
            "这段文本共有 10 个字符。",
        ),
        {
            "expected_tool_sequence": ["text_length"],
            "expected_args": {"text_length": {"text": "OpenAI API"}},
            "expected_obs_keys": {"text_length": []},
            "expected_answer": "10",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert info["obs_use_acc"]
    assert info["answer_acc"]


def test_hard_multistep_trajectory_gets_high_reward():
    traj = ToolTrajectory(
        task_id="hard_tm_test",
        prompt="统计 OpenAI API 的字符数，然后把字符数乘以3",
        tools=["text_length", "calculate_math"],
        steps=[
            TrajectoryStep(role="user", type="user", content="统计 OpenAI API 的字符数，然后把字符数乘以3"),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": "text_length", "arguments": {"text": "OpenAI API"}},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": "text_length", "arguments": {"text": "OpenAI API"}, "result": {"text": "OpenAI API", "characters": 10, "words": 2}},
            ),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "10*3"}},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "10*3"}, "result": {"expression": "10*3", "result": 30}},
            ),
            TrajectoryStep(role="assistant", type="assistant_final", content="计算结果是 30。"),
        ],
        final_answer="计算结果是 30。",
        done=True,
    )
    reward, info = compute_total_reward(
        traj,
        {
            "expected_tool_sequence": ["text_length", "calculate_math"],
            "expected_args": {"text_length": {"text": "OpenAI API"}, "calculate_math": {"expression": "10*3"}},
            "expected_obs_keys": {"text_length": ["characters"], "calculate_math": ["result"]},
            "expected_answer": "30",
            "expected_answer_type": "number",
            "max_tool_calls": 2,
        },
    )
    assert info["tool_acc"]
    assert info["args_acc"]
    assert info["obs_use_acc"]
    assert info["answer_acc"]
    assert reward > 2.0


def test_tool_call_without_final_is_failure():
    traj = ToolTrajectory(
        task_id="no_final",
        prompt="计算 9 的平方根",
        tools=["calculate_math"],
        steps=[
            TrajectoryStep(role="user", type="user", content="计算 9 的平方根"),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "sqrt(9)"}},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "sqrt(9)"}, "result": {"expression": "sqrt(9)", "result": 3}},
            ),
        ],
        final_answer="",
        done=True,
    )
    _, info = compute_total_reward(
        traj,
        {
            "expected_tool_sequence": ["calculate_math"],
            "expected_args": {"calculate_math": {"expression": "sqrt(9)"}},
            "expected_answer": "3",
            "expected_answer_type": "number",
            "max_tool_calls": 1,
        },
    )
    assert info["tool_call_without_final"]
    assert "tool_call_without_final" in info["failures"]


def test_reward_penalizes_loop():
    env = ToolUseEnv(mode="guarded")
    env.reset({"id": "loop_001", "prompt": "算 1+1", "tools": ["calculate_math"]})
    output = '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
    env.step(output)
    env.step(output)
    reward, info = compute_total_reward(env.get_trajectory(), {"expected_tool_sequence": ["calculate_math"], "expected_answer": "2"})
    assert info["loop"]
    assert reward < 2.0


def test_reward_penalizes_unnecessary_tool_for_no_tool_task():
    task = {
        "id": "notool_001",
        "tools": ["calculate_math", "text_length"],
        "expected_tool_sequence": [],
        "expected_args": {},
        "args_match": "none",
        "max_tool_calls": 0,
        "allow_no_tool": True,
    }
    good_reward, good_info = compute_total_reward(build_final_traj("平方根是一个数自乘后得到原数的值。"), task)
    bad_reward, bad_info = compute_total_reward(
        build_single_tool_traj(
            "calculate_math",
            {"expression": "sqrt(4)"},
            {"expression": "sqrt(4)", "result": 2.0},
            "平方根是一个数自乘后得到原数的值。",
        ),
        task,
    )
    assert "unnecessary_tool" in bad_info["failures"]
    assert bad_info["unnecessary_tool_penalty"] == -0.5
    assert good_info["unnecessary_tool_penalty"] == 0.0
    assert good_reward > bad_reward


def build_repeated_math_traj(final_answer="5"):
    return ToolTrajectory(
        task_id="dense_repeat",
        prompt="first compute 1+1, then compute 2+3",
        tools=["calculate_math"],
        steps=[
            TrajectoryStep(role="user", type="user", content="first compute 1+1, then compute 2+3"),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "1+1"}},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "1+1"}, "result": {"expression": "1+1", "result": 2}},
            ),
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "2+3"}},
            ),
            TrajectoryStep(
                role="tool",
                type="tool_observation",
                content="",
                meta={"name": "calculate_math", "arguments": {"expression": "2+3"}, "result": {"expression": "2+3", "result": 5}},
            ),
            TrajectoryStep(role="assistant", type="assistant_final", content=final_answer),
        ],
        final_answer=final_answer,
        done=True,
    )


def test_repeated_tool_expected_args_are_matched_by_occurrence():
    task = {
        "expected_tool_sequence": ["calculate_math", "calculate_math"],
        "expected_args": {"calculate_math": [{"expression": "1+1"}, {"expression": "2+3"}]},
        "expected_obs_keys": {"calculate_math": ["result"]},
        "expected_answer": "5",
        "expected_answer_type": "number",
        "max_tool_calls": 2,
    }

    _, good_info = compute_total_reward(build_repeated_math_traj(), task)
    _, bad_info = compute_total_reward(
        build_repeated_math_traj(),
        {**task, "expected_args": {"calculate_math": [{"expression": "2+3"}, {"expression": "1+1"}]}},
    )

    assert good_info["args_acc"]
    assert not bad_info["args_acc"]


def test_dense_v2_correct_trajectory_is_high_but_not_saturated():
    task = {
        "expected_tool_sequence": ["calculate_math", "calculate_math"],
        "expected_args": {"calculate_math": [{"expression": "1+1"}, {"expression": "2+3"}]},
        "expected_obs_keys": {"calculate_math": ["result"]},
        "expected_answer": "5",
        "expected_answer_type": "number",
        "max_tool_calls": 2,
        "reward_profile": "dense_v2",
    }

    reward, info = compute_total_reward(build_repeated_math_traj("after 2, final result is 5"), task)

    assert info["reward_profile"] == "dense_v2"
    assert info["answer_acc"]
    assert 2.0 < reward < 3.0


def test_dense_v2_penalizes_extra_tool_call():
    task = {
        "expected_tool_sequence": ["calculate_math"],
        "expected_args": {"calculate_math": {"expression": "2+3"}},
        "expected_obs_keys": {"calculate_math": ["result"]},
        "expected_answer": "5",
        "expected_answer_type": "number",
        "max_tool_calls": 1,
        "reward_profile": "dense_v2",
    }
    good = build_single_tool_traj("calculate_math", {"expression": "2+3"}, {"expression": "2+3", "result": 5}, "5")
    extra = build_repeated_math_traj("5")

    good_reward, _ = compute_total_reward(good, task)
    extra_reward, extra_info = compute_total_reward(extra, task)

    assert "overuse_tool" in extra_info["failures"]
    assert good_reward > extra_reward


def test_dense_v2_rewards_grounded_answer_over_ungrounded_correct_answer():
    task = {
        "expected_tool_sequence": ["calculate_math"],
        "expected_args": {"calculate_math": {"expression": "1+2"}},
        "expected_obs_keys": {"calculate_math": ["result"]},
        "expected_answer": "12",
        "expected_answer_type": "number",
        "max_tool_calls": 1,
        "reward_profile": "dense_v2",
    }
    ungrounded = build_single_tool_traj("calculate_math", {"expression": "1+2"}, {"expression": "1+2", "result": 3}, "12")
    grounded = build_single_tool_traj("calculate_math", {"expression": "1+2"}, {"expression": "1+2", "result": 3}, "tool result 3, final answer 12")

    ungrounded_reward, ungrounded_info = compute_total_reward(ungrounded, task)
    grounded_reward, grounded_info = compute_total_reward(grounded, task)

    assert ungrounded_info["answer_acc"]
    assert not ungrounded_info["obs_use_strict_acc"]
    assert grounded_info["obs_use_strict_acc"]
    assert grounded_reward > ungrounded_reward


def test_dense_v2_no_tool_task_penalizes_unnecessary_tool():
    task = {
        "expected_tool_sequence": [],
        "expected_args": {},
        "args_match": "none",
        "max_tool_calls": 0,
        "allow_no_tool": True,
        "reward_profile": "dense_v2",
    }
    good_reward, _ = compute_total_reward(build_final_traj("Answer directly without a tool."), task)
    bad_reward, bad_info = compute_total_reward(
        build_single_tool_traj("calculate_math", {"expression": "2+2"}, {"expression": "2+2", "result": 4}, "Answer directly without a tool."),
        task,
    )

    assert "unnecessary_tool" in bad_info["failures"]
    assert good_reward > bad_reward
