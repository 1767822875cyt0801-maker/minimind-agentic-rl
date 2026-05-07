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
