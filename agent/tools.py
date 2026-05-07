#负责工具注册和工具执行
import ast
import math
import operator as op
import random
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


ToolArgs = Dict[str, Any]
ToolResult = Dict[str, Any]


ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}


def normalize_expression(expr: Any) -> str:
    return (
        str(expr)
        .replace("math.sqrt", "sqrt")
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("²", "**2")
        .replace("³", "**3")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
    )


def safe_eval_expr(expr: Any) -> float:
    expr = normalize_expression(expr)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"operator {op_type.__name__} is not allowed")
            return ALLOWED_OPERATORS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"unary operator {op_type.__name__} is not allowed")
            return ALLOWED_OPERATORS[op_type](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function calls are allowed")
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCS:
                raise ValueError(f"function {func_name} is not allowed")
            return ALLOWED_FUNCS[func_name](*[_eval(arg) for arg in node.args])
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return _eval(ast.parse(expr, mode="eval"))


def calculate_math(args: ToolArgs) -> ToolResult:
    expression = normalize_expression(args.get("expression", "0"))
    return {"expression": expression, "result": safe_eval_expr(expression)}


def get_current_time(args: ToolArgs) -> ToolResult:
    timezone = str(args.get("timezone") or "Asia/Dalian")
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        timezone = "Asia/Shanghai"
        now = datetime.now(ZoneInfo(timezone))
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
        "weekday": now.strftime("%A"),
    }


def random_number(args: ToolArgs) -> ToolResult:
    min_value = int(args.get("min", 0))
    max_value = int(args.get("max", 100))
    if min_value > max_value:
        min_value, max_value = max_value, min_value
    return {"min": min_value, "max": max_value, "result": random.randint(min_value, max_value)}


def text_length(args: ToolArgs) -> ToolResult:
    text = str(args.get("text", ""))
    return {"text": text, "characters": len(text), "words": len(text.split())}


def unit_converter(args: ToolArgs) -> ToolResult:
    value = float(args.get("value", 0))
    from_unit = str(args.get("from_unit", "")).lower()
    to_unit = str(args.get("to_unit", "")).lower()

    length_units_to_meter = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "mile": 1609.344,
        "miles": 1609.344,
        "ft": 0.3048,
        "feet": 0.3048,
    }
    weight_units_to_kg = {
        "kg": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "g": 0.001,
        "gram": 0.001,
        "grams": 0.001,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "lb": 0.45359237,
        "lbs": 0.45359237,
    }

    if from_unit in ["celsius", "℃", "c"] and to_unit in ["fahrenheit", "℉", "f"]:
        result = value * 9 / 5 + 32
    elif from_unit in ["fahrenheit", "℉", "f"] and to_unit in ["celsius", "℃", "c"]:
        result = (value - 32) * 5 / 9
    elif from_unit in length_units_to_meter and to_unit in length_units_to_meter:
        result = value * length_units_to_meter[from_unit] / length_units_to_meter[to_unit]
    elif from_unit in weight_units_to_kg and to_unit in weight_units_to_kg:
        result = value * weight_units_to_kg[from_unit] / weight_units_to_kg[to_unit]
    else:
        return {"error": f"unsupported conversion: {from_unit} -> {to_unit}"}

    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 6),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果，支持加减乘除、幂运算、平方根和常见数学函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 256*37、2**10、sqrt(144)",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，支持 IANA 时区名称",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区名称，如 Asia/Shanghai、America/New_York",
                        "default": "Asia/Shanghai",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "生成指定闭区间内的随机整数",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "integer", "description": "最小值", "default": 0},
                    "max": {"type": "integer", "description": "最大值", "default": 100},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_length",
            "description": "统计文本字符数和按空白切分的单词数",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "要统计的文本"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "进行单位换算，支持长度、重量、摄氏度和华氏度",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "要转换的数值"},
                    "from_unit": {"type": "string", "description": "源单位，如 km、miles、kg、pounds、celsius"},
                    "to_unit": {"type": "string", "description": "目标单位"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "calculate_math": {"schema": TOOLS[0], "func": calculate_math},
    "get_current_time": {"schema": TOOLS[1], "func": get_current_time},
    "random_number": {"schema": TOOLS[2], "func": random_number},
    "text_length": {"schema": TOOLS[3], "func": text_length},
    "unit_converter": {"schema": TOOLS[4], "func": unit_converter},
}


def get_tool_name(schema_or_name: Any) -> str:
    if isinstance(schema_or_name, str):
        return schema_or_name
    if isinstance(schema_or_name, dict):
        fn = schema_or_name.get("function", {})
        return fn.get("name", "")
    return ""


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def get_tool_schemas(names: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    if names is None:
        return get_all_tool_schemas()
    schemas = []
    for item in names:
        if isinstance(item, dict):
            schemas.append(item)
            continue
        entry = TOOL_REGISTRY.get(str(item))
        if entry:
            schemas.append(entry["schema"])
    return schemas


def get_tool_names(tools: Optional[Iterable[Any]] = None) -> List[str]:
    if tools is None:
        return list(TOOL_REGISTRY.keys())
    return [name for item in tools if (name := get_tool_name(item))]


def validate_tool_args(name: str, args: ToolArgs) -> bool:
    if not isinstance(args, dict):
        return False
    if name == "calculate_math":
        return bool(args.get("expression"))
    if name == "get_current_time":
        return True
    if name == "random_number":
        return True
    if name == "text_length":
        return "text" in args
    if name == "unit_converter":
        return args.get("value") is not None and bool(args.get("from_unit")) and bool(args.get("to_unit"))
    return False


def execute_tool_call(name: str, arguments: Optional[ToolArgs] = None) -> ToolResult:
    args = arguments or {}
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"error": f"unknown tool: {name}", "available_tools": list(TOOL_REGISTRY.keys())}
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object", "raw_arguments": str(args)}
    try:
        return entry["func"](args)
    except Exception as exc:
        return {"error": str(exc)[:300]}


def get_tool_function(name: str) -> Optional[Callable[[ToolArgs], ToolResult]]:
    entry = TOOL_REGISTRY.get(name)
    return entry["func"] if entry else None

