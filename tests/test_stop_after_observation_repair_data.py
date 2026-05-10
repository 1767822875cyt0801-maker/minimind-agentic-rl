import json

from scripts.make_tool_sft_stop_after_observation_repair import build_rows, summarize


def test_stop_after_observation_repair_rows_have_expected_shape():
    rows = build_rows(40)

    assert len(rows) == 40
    assert set(summarize(rows)) == {
        "stop_repair_math",
        "stop_repair_text",
        "stop_repair_time",
        "stop_repair_unit",
        "stop_repair_no_tool",
    }

    prompts = [row["conversations"][1]["content"] for row in rows]
    assert len(prompts) == len(set(prompts))

    tool_rows = [row for row in rows if row["category"] != "stop_repair_no_tool"]
    assert tool_rows
    for row in tool_rows:
        conversations = row["conversations"]
        assert [message["role"] for message in conversations] == ["system", "user", "assistant", "tool", "assistant"]
        assert json.loads(conversations[0]["tools"])
        assert json.loads(conversations[2]["tool_calls"])[0]["type"] == "function"
        assert conversations[-1]["content"].strip()


def test_stop_after_observation_repair_no_tool_rows_keep_tools_available():
    rows = build_rows(40)
    no_tool_rows = [row for row in rows if row["category"] == "stop_repair_no_tool"]

    assert no_tool_rows
    for row in no_tool_rows:
        conversations = row["conversations"]
        assert [message["role"] for message in conversations] == ["system", "user", "assistant"]
        assert json.loads(conversations[0]["tools"])
        assert "tool_calls" not in conversations[-1]
