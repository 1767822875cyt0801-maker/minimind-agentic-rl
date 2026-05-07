from agent.tools import execute_tool_call, get_tool_schemas


def test_calculate_math_executes_safely():
    result = execute_tool_call("calculate_math", {"expression": "256*37"})
    assert result["result"] == 9472


def test_unit_converter_reports_unsupported_conversion():
    result = execute_tool_call("unit_converter", {"value": 1, "from_unit": "km", "to_unit": "kg"})
    assert "error" in result


def test_tool_schema_lookup_by_name():
    schemas = get_tool_schemas(["calculate_math"])
    assert schemas[0]["function"]["name"] == "calculate_math"

