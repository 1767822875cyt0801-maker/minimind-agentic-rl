from agent.tool_env import ToolUseEnv


def test_tool_env_executes_tool_and_records_trajectory():
    env = ToolUseEnv(mode="raw")
    env.reset({"id": "math_001", "prompt": "算 256*37", "tools": ["calculate_math"]})
    info = env.step('<tool_call>{"name":"calculate_math","arguments":{"expression":"256*37"}}</tool_call>')
    traj = env.get_trajectory()
    assert info["type"] == "tool_observation"
    assert info["observation"]["result"] == 9472
    assert traj.steps[-1].type == "tool_observation"


def test_guarded_env_blocks_repeated_tool_call():
    env = ToolUseEnv(mode="guarded", max_tool_calls=3)
    env.reset({"id": "loop_001", "prompt": "算 1+1", "tools": ["calculate_math"]})
    output = '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
    env.step(output)
    info = env.step(output)
    assert info["type"] == "runtime_guard"
    assert info["done"]
    assert env.get_trajectory().metrics["loop_count"] == 1

