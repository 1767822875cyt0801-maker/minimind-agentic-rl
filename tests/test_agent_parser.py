from agent.parser import parse_first_tool_call, parse_tool_calls


def test_parse_valid_tool_call():
    text = '<tool_call>{"name":"calculate_math","arguments":{"expression":"2**8"}}</tool_call>'
    result = parse_first_tool_call(text)
    assert result.ok
    assert result.tool_name == "calculate_math"
    assert result.arguments == {"expression": "2**8"}


def test_parse_bad_json_in_raw_mode():
    result = parse_first_tool_call('<tool_call>{"name":"calculate_math",</tool_call>', strict=True)
    assert not result.ok
    assert result.error_type == "malformed_json"


def test_guarded_mode_repairs_single_quotes():
    result = parse_first_tool_call("<tool_call>{'name':'text_length','arguments':{'text':'abc'}}</tool_call>", strict=False)
    assert result.ok
    assert result.json_repaired


def test_parse_multiple_tool_calls():
    text = (
        '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
        '<tool_call>{"name":"text_length","arguments":{"text":"abc"}}</tool_call>'
    )
    results = parse_tool_calls(text)
    assert [r.tool_name for r in results if r.ok] == ["calculate_math", "text_length"]

