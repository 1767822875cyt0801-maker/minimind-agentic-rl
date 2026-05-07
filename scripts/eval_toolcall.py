import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from models.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')
import ast
import operator as op
import math
from pathlib import Path
from agent.tool_env import ToolUseEnv

#5.3 加一个清洗函数
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

#安全数学计算工具，基于AST的安全表达式求值
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

def safe_eval_expr(expr: str):
    expr = (
        str(expr)
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("²", "**2")
        .replace("³", "**3")
        .replace("（", "(")
        .replace("）", ")")
    )

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Operator {op_type.__name__} is not allowed")
            return ALLOWED_OPERATORS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
            return ALLOWED_OPERATORS[op_type](_eval(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCS:
                raise ValueError(f"Function {func_name} is not allowed")
            args = [_eval(arg) for arg in node.args]
            return ALLOWED_FUNCS[func_name](*args)

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def calculate_math(args):
    expression = args.get("expression", "0")
    result = safe_eval_expr(expression)
    return {
        "expression": expression,
        "result": result
    }

#当前时间工具
from zoneinfo import ZoneInfo

def get_current_time(args):
    timezone = args.get("timezone", "Asia/Dalian")

    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        timezone = "Asia/Dalian"
        now = datetime.now(ZoneInfo(timezone))

    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
        "weekday": now.strftime("%A")
    }


#随机数工具
import random

def random_number_tool(args):
    min_value = int(args.get("min", 0))
    max_value = int(args.get("max", 100))

    if min_value > max_value:
        min_value, max_value = max_value, min_value

    value = random.randint(min_value, max_value)

    return {
        "min": min_value,
        "max": max_value,
        "result": value
    }

#修改通用单位转换功能
def unit_converter(args):
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

    # 温度换算
    if from_unit in ["celsius", "℃", "c"] and to_unit in ["fahrenheit", "℉", "f"]:
        result = value * 9 / 5 + 32
    elif from_unit in ["fahrenheit", "℉", "f"] and to_unit in ["celsius", "℃", "c"]:
        result = (value - 32) * 5 / 9

    # 长度换算
    elif from_unit in length_units_to_meter and to_unit in length_units_to_meter:
        meter_value = value * length_units_to_meter[from_unit]
        result = meter_value / length_units_to_meter[to_unit]

    # 重量换算
    elif from_unit in weight_units_to_kg and to_unit in weight_units_to_kg:
        kg_value = value * weight_units_to_kg[from_unit]
        result = kg_value / weight_units_to_kg[to_unit]

    else:
        return {
            "error": f"Unsupported conversion: {from_unit} -> {to_unit}"
        }

    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 6)
    }

def text_length(args):
    text = args.get("text", "")

    return {
        "text": text,
        "characters": len(text),
        "words": len(text.split())
    }


def clean_terminal_text(s):
    if not isinstance(s, str):
        return s
    # 删除方向键、控制键等 ANSI escape sequence，例如 \x1b[D
    s = ANSI_ESCAPE_RE.sub('', s)
    # 再删除残留 ESC
    s = s.replace('\x1b', '')
    return s.strip()

def clean_terminal_text(s):
    if not isinstance(s, str):
        return s
    # 删除方向键、控制键等 ANSI escape sequence，例如 \x1b[D
    s = ANSI_ESCAPE_RE.sub('', s)
    # 再删除残留 ESC
    s = s.replace('\x1b', '')
    return s.strip()

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Dalian、America/New_York", "default": "Asia/Dalian"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    # {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    # {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    # {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

REAL_TOOL_FUNCS = {
    "calculate_math": calculate_math,
    "get_current_time": get_current_time,
    "random_number": random_number_tool,
    "text_length": text_length,
    "unit_converter": unit_converter,
    # 下面三个可以先不接，或者后续接 API
    # "get_current_weather": get_current_weather,
    # "get_exchange_rate": get_exchange_rate,
    # "translate_text": translate_text,
}


# MOCK_RESULTS = {
#     "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
#     "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Dalian")},
#     "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
#     "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
#     "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
#     "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
#     "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
#     "translate_text": lambda args: {"translated": "hello world"},
# }

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    project_root = Path(__file__).resolve().parents[1]
    load_from = Path(args.load_from)
    if not load_from.is_absolute():
        load_from = project_root / load_from
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir

    tokenizer = AutoTokenizer.from_pretrained(str(load_from))
    if load_from.name.startswith('model'):
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = save_dir / f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(str(ckp), map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(load_from), trust_remote_code=True)
    get_model_params(model, model.config)
    model = model.eval().to(args.device)
    if "cuda" in str(args.device):
        model = model.half()
    return model, tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def parse_arguments(raw_args):
    if raw_args is None:
        return {}

    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        return json.loads(raw_args)

    raise ValueError(f"Unsupported arguments type: {type(raw_args)}")


def execute_tool(call, arguments=None):
    """
    真实工具执行入口。

    local 模式:
        call = {"name": "...", "arguments": {...}}

    api 模式:
        call = "tool_name"
        arguments = '{"...": "..."}'
    """
    if isinstance(call, dict):
        name = call.get("name", "")
        raw_args = call.get("arguments", {})
    else:
        name = call
        raw_args = arguments

    try:
        args = parse_arguments(raw_args)
    except Exception as e:
        return {
            "error": "Invalid tool arguments",
            "detail": str(e),
            "raw_arguments": str(raw_args)
        }

    fn = REAL_TOOL_FUNCS.get(name)

    if fn is None:
        return {
            "error": f"Unknown tool: {name}",
            "available_tools": list(REAL_TOOL_FUNCS.keys())
        }

    try:
        result = fn(args)
        return result
    except Exception as e:
        return {
            "error": str(e)[:300]
        }


def generate(model, tokenizer, messages, tools, args, stream_output=True):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream_output else None
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools if tools else None, open_thinking=False)

    # 5.3 修改generate()，检查input_ids是否是str。在出错时，能判断是input_text类型问题还是tokenizer编码问题
    if not isinstance(input_text, str):
        raise TypeError(
            f"apply_chat_template should return str, got {type(input_text)}: {repr(input_text)[:500]}"
        )
    input_text = clean_terminal_text(input_text)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    if stream_output:
        print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=False, streamer=streamer,# tool calling 建议先关掉随机采样
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    if stream_output:
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def generate_final_answer(model, tokenizer, messages, args):
    """
    工具已经执行完之后，强制模型根据 tool result 生成最终答案。
    这里不再传 tools，避免模型继续调用工具。
    注意：部分小模型仍可能继续生成 <tool_call>，所以本函数只返回原始文本，
    是否展示给用户由 run_case 里的 final formatter 决定。
    """
    final_messages = messages + [{
        "role": "user",
        "content": (
            "请根据上面的工具返回结果，直接回答我的原始问题。"
            "不要再输出 <tool_call>，不要再调用任何工具。"
        )
    }]

    input_text = tokenizer.apply_chat_template(
        final_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        open_thinking=False
    )

    if not isinstance(input_text, str):
        raise TypeError(
            f"apply_chat_template should return str, got {type(input_text)}: {repr(input_text)[:500]}"
        )

    input_text = clean_terminal_text(input_text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    ).to(args.device)

    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=min(args.max_new_tokens, 256),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )

    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls

def normalize_arguments_for_key(arguments):
    try:
        args = parse_arguments(arguments)
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(arguments)


def format_scalar(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def clean_model_final_text(text):
    if not isinstance(text, str):
        return ""

    text = clean_terminal_text(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<tool_response>.*?</tool_response>', '', text, flags=re.DOTALL).strip()

    if "<tool_call>" in text or "</tool_call>" in text:
        return ""
    return text


def has_sqrt_intent(prompt):
    prompt = str(prompt).lower()
    return any(keyword in prompt for keyword in ["平方根", "开方", "根号", "sqrt", "square root"])


def has_square_intent(prompt):
    prompt = str(prompt).lower()
    if has_sqrt_intent(prompt):
        return False
    return any(keyword in prompt for keyword in ["平方", "square"])


def expression_has_sqrt(expression):
    expression = str(expression).replace(" ", "").lower()
    return any(mark in expression for mark in ["sqrt(", "**0.5", "^0.5"])


def expression_is_square_of_value(expression, value):
    expression = str(expression).replace(" ", "").lower()
    value = format_scalar(value)
    return any(pattern in expression for pattern in [
        f"{value}**2",
        f"{value}^2",
        f"{value}*{value}",
        f"pow({value},2)",
    ])


def latest_tool_result(tool_history, name):
    for item in reversed(tool_history):
        if item["name"] == name and isinstance(item["result"], dict):
            return item["result"]
    return None


def has_tool_history(tool_history, name):
    return any(item["name"] == name for item in tool_history)


def maybe_fix_tool_call_arguments(prompt, tool_call, tool_history):
    if tool_call["name"] != "calculate_math" or not has_sqrt_intent(prompt):
        return tool_call, None

    try:
        parsed_args = parse_arguments(tool_call["arguments"])
    except Exception:
        return tool_call, None

    expression = parsed_args.get("expression", "")
    random_result = latest_tool_result(tool_history, "random_number")
    random_value = random_result.get("result") if random_result else None

    if random_value is None or expression_has_sqrt(expression):
        return tool_call, None

    if expression_is_square_of_value(expression, random_value):
        fixed_args = dict(parsed_args)
        fixed_args["expression"] = f"sqrt({format_scalar(random_value)})"
        fixed_call = dict(tool_call)
        fixed_call["arguments"] = fixed_args
        return fixed_call, f"{expression} -> {fixed_args['expression']}"

    return tool_call, None


def build_missing_followup_tool_call(prompt, tool_history):
    if has_tool_history(tool_history, "calculate_math"):
        return None

    random_result = latest_tool_result(tool_history, "random_number")
    random_value = random_result.get("result") if random_result else None
    if random_value is None:
        return None

    if has_sqrt_intent(prompt):
        expression = f"sqrt({format_scalar(random_value)})"
    elif has_square_intent(prompt):
        expression = f"{format_scalar(random_value)}**2"
    else:
        return None

    return {
        "id": "auto_calculate_math",
        "name": "calculate_math",
        "arguments": {
            "expression": expression
        }
    }


def render_tool_calls_content(tool_calls):
    blocks = []
    for tc in tool_calls:
        args = parse_arguments(tc["arguments"])
        data = {
            "name": tc["name"],
            "arguments": args
        }
        blocks.append("<tool_call>\n" + json.dumps(data, ensure_ascii=False) + "\n</tool_call>")
    return "\n\n".join(blocks)


def format_tool_answer(name, arguments, result):
    try:
        parsed_args = parse_arguments(arguments)
    except Exception:
        parsed_args = {}

    if not isinstance(result, dict):
        return f"工具返回结果：{result}"

    if "error" in result:
        detail = result.get("detail")
        return f"工具执行失败：{result['error']}" + (f"（{detail}）" if detail else "")

    if name == "calculate_math":
        expression = result.get("expression") or parsed_args.get("expression", "")
        value = result.get("result")
        return f"{expression} = {format_scalar(value)}" if expression else format_scalar(value)

    if name == "get_current_time":
        dt = result.get("datetime", "")
        timezone = result.get("timezone", "")
        weekday = result.get("weekday", "")
        suffix = "，".join([x for x in [timezone, weekday] if x])
        return f"当前时间是 {dt}" + (f"（{suffix}）" if suffix else "")

    if name == "random_number":
        value = result.get("result")
        min_value = result.get("min")
        max_value = result.get("max")
        if min_value is not None and max_value is not None:
            return f"生成的随机数是 {value}（范围：{min_value} 到 {max_value}）"
        return f"生成的随机数是 {value}"

    if name == "text_length":
        return f"这段文本共有 {result.get('characters')} 个字符，{result.get('words')} 个单词"

    if name == "unit_converter":
        value = format_scalar(result.get("value"))
        from_unit = result.get("from_unit", "")
        converted = format_scalar(result.get("result"))
        to_unit = result.get("to_unit", "")
        return f"{value} {from_unit} = {converted} {to_unit}"

    return json.dumps(result, ensure_ascii=False)


def format_tool_history_answer(tool_history, original_prompt=""):
    if not tool_history:
        return ""

    if len(tool_history) == 1:
        item = tool_history[0]
        return format_tool_answer(item["name"], item["arguments"], item["result"])

    last = tool_history[-1]
    if last["name"] == "calculate_math" and isinstance(last["result"], dict):
        random_items = [
            item for item in tool_history
            if item["name"] == "random_number" and isinstance(item["result"], dict)
        ]
        expression = last["result"].get("expression") or ""
        calc_value = format_scalar(last["result"].get("result"))
        sqrt_intent = has_sqrt_intent(original_prompt) or expression_has_sqrt(expression)

        if random_items:
            random_value = format_scalar(random_items[-1]["result"].get("result"))
            if sqrt_intent:
                return f"生成的随机数是 {random_value}，它的平方根是 {calc_value}"
            square_marks = ["**2", "^2", f"{random_value}*{random_value}", f"{random_value} * {random_value}"]
            if any(mark in expression.replace(" ", "") for mark in square_marks[:3]) or square_marks[3] in expression:
                return f"生成的随机数是 {random_value}，它的平方是 {calc_value}"
            return f"生成的随机数是 {random_value}，计算结果是 {calc_value}"

        if sqrt_intent:
            return f"平方根是 {calc_value}"
        return format_tool_answer(last["name"], last["arguments"], last["result"])

    return "；".join(
        format_tool_answer(item["name"], item["arguments"], item["result"])
        for item in tool_history
    )


def print_final_answer(name=None, arguments=None, result=None, model_text="", tool_history=None, original_prompt=""):
    answer = clean_model_final_text(model_text)
    if not answer:
        if tool_history:
            answer = format_tool_history_answer(tool_history, original_prompt)
        else:
            answer = format_tool_answer(name, arguments, result)
    print(f"🧠 Final: {answer}")


def normalize_tool_calls(tool_calls):
    normalized = []
    for i, tc in enumerate(tool_calls or []):
        if isinstance(tc, dict):
            if "name" in tc:
                name = tc.get("name", "")
                arguments = tc.get("arguments", {})
            else:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                arguments = fn.get("arguments", {})
            call_id = tc.get("id") or f"call_{i}"
        else:
            fn = tc.function
            name = fn.name
            arguments = fn.arguments
            call_id = tc.id or f"call_{i}"

        normalized.append({
            "id": call_id,
            "name": name,
            "arguments": arguments
        })
    return normalized


def append_assistant_tool_message(messages, content, tool_calls, backend):
    if backend == 'api':
        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"], ensure_ascii=False)
                    }
                }
                for tc in tool_calls
            ]
        })
        return

    messages.append({
        "role": "assistant",
        "content": content
    })


def append_tool_result_message(messages, tool_call, result, backend):
    message = {
        "role": "tool",
        "content": json.dumps(result, ensure_ascii=False)
    }
    if backend == 'api':
        message["tool_call_id"] = tool_call["id"]
    messages.append(message)


# 增加工具调用轮数限制：允许多轮工具调用，同时避免重复调用陷入循环
def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    env = ToolUseEnv(mode=args.mode, max_tool_calls=args.max_tool_rounds)
    env.reset({"id": "interactive", "prompt": clean_terminal_text(prompt), "tools": tools})

    for tool_round in range(args.max_tool_rounds + 1):
        if args.backend == 'local':
            content = generate(model, tokenizer, env.get_messages(), env.get_tool_schemas() or tools, args, stream_output=(tool_round == 0))
        else:
            content, tool_calls = chat_api(client, env.get_messages(), env.get_tool_schemas() or tools, args, stream=bool(args.stream))
            if tool_calls:
                content = render_tool_calls_content(normalize_tool_calls(tool_calls))

        if args.backend == 'local' and tool_round > 0:
            print(f'🧠: {content}')

        info = env.step(content)
        if info.get("type") == "tool_observation":
            parse = info.get("parse")
            name = parse.tool_name if parse else ""
            arguments = parse.arguments if parse else {}
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            print(f'✅ [Tool Called]: {json.dumps(info.get("observation"), ensure_ascii=False)}')
            continue

        if info.get("type") == "parse_error":
            parse = info.get("parse")
            print(f"⚠️ 工具调用解析失败：{parse.error_type if parse else 'unknown'}")

        if info.get("done"):
            print(f"🧠 Final: {env.render_final_answer()}")
            return

    env.mark_unfinished()
    print("⚠️ 达到最大工具调用轮数，停止继续调用工具。")
    print(f"🧠 Final: {env.render_final_answer()}")


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='models', type=str, help="模型加载路径（models=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    parser.add_argument('--mode', default='guarded', choices=['raw', 'guarded'], type=str, help="工具环境模式")
    parser.add_argument('--max_tool_rounds', default=4, type=int, help="最大工具调用轮数")
    args = parser.parse_args()

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # 5.3 改手动输入逻辑
    def manual_cases():
        while True:
            prompt = clean_terminal_text(input('💬: '))
            if not prompt:
                break
            yield {
                "prompt": prompt,
                "tools": TOOLS,
                "tool_names": [t["function"]["name"] for t in TOOLS]
            }
    
    if input_mode == 0:
        cases = [
            {
                "prompt": case["prompt"],
                "tools": get_tools(case["tools"]),
                "tool_names": case["tools"]
            }
            for case in TEST_CASES
        ]
    else:
        cases = manual_cases()
    # cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
