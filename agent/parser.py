#负责解析模型输出中的 <tool_call>{"name": "...", "arguments": {...}}</tool_call>

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

#匹配 ANSI 转义序列（终端控制码）
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
#提取 XML 标签内的工具调用。从模型生成的文本中，提取被 <tool_call> 和 </tool_call> 包裹的内容。
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ToolCallParseResult:
    ok: bool
    raw_text: str
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    json_repaired: bool = False


def clean_terminal_text(text: Any) -> str:
    if not isinstance(text, str):
        return str(text)
    return ANSI_ESCAPE_RE.sub("", text).replace("\x1b", "").strip()

#从模型生成的文本中提取第一个工具调用的内容
def _extract_first_block(text: str, strict: bool) -> Optional[str]:
    #用正则尝试匹配标准标签，正则默认非贪婪，所以返回第一个<tool_call>标签对的内容
    match = TOOL_CALL_RE.search(text)
    if match:
        return match.group(1).strip()
    #严格模式快速返回 None
    if strict or "<tool_call>" not in text:
        return None
    #非严格模式下的手工提取（容错）
    after = text.split("<tool_call>", 1)[1]
    if "</tool_call>" in after:
        after = after.split("</tool_call>", 1)[0]
    if "{" in after and "}" in after:
        return after[after.find("{") : after.rfind("}") + 1].strip()
    return after.strip()

#修复模型生成的原始文本，使其成为可解析的 JSON 字符串
def _repair_json_text(raw: str) -> str:
    text = clean_terminal_text(raw)
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

#安全地将模型输出的原始字符串解析为 JSON 对象（字典），支持严格模式和宽松模式，并包含自动修复功能
def _loads_json(raw: str, strict: bool) -> tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    try:
        return json.loads(raw), False, None
    except Exception as first_exc:
        if strict:
            return None, False, str(first_exc)

    repaired = _repair_json_text(raw)
    try:
        return json.loads(repaired), repaired != raw, None
    except Exception:
        pass

    try:
        #用 ast.literal_eval 兜底解析
        value = ast.literal_eval(repaired)
        if isinstance(value, dict):
            return value, True, None
    except Exception as second_exc:
        return None, repaired != raw, str(second_exc)

    return None, repaired != raw, "tool call JSON must decode to an object"


def normalize_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        return json.loads(raw_args)
    raise ValueError(f"unsupported arguments type: {type(raw_args).__name__}")


def normalize_tool_call_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if "function" in data and isinstance(data["function"], dict):
        fn = data["function"]
        return {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
    return {"name": data.get("name", ""), "arguments": data.get("arguments", {})}


def parse_first_tool_call(
    text: str,
    strict: bool = True,
    known_tools: Optional[Iterable[str]] = None,
) -> ToolCallParseResult:
    #清理终端控制字符
    raw_text = clean_terminal_text(text)
    #提取第一个工具调用块
    raw_block = _extract_first_block(raw_text, strict=strict)
    #处理提取失败（没有找到工具调用块）
    if raw_block is None:
        if "<tool_call>" in raw_text or "</tool_call>" in raw_text:
            return ToolCallParseResult(False, raw_text, error_type="malformed_tags", error_msg="unbalanced tool_call tags")
        return ToolCallParseResult(False, raw_text, error_type="no_tool_call", error_msg="no tool_call block found")
    #解析 JSON（含自动修复）
    data, repaired, err = _loads_json(raw_block, strict=strict)
    if data is None:
        return ToolCallParseResult(False, raw_text, error_type="malformed_json", error_msg=err, json_repaired=repaired)
    # 验证解析结果必须是字典
    if not isinstance(data, dict):
        return ToolCallParseResult(False, raw_text, error_type="malformed_json", error_msg="tool call must be a JSON object")
    #提取工具名称并检查
    normalized = normalize_tool_call_dict(data)
    name = normalized.get("name") or ""
    if not name:
        return ToolCallParseResult(False, raw_text, error_type="missing_name", error_msg="missing tool name", json_repaired=repaired)

    try:
        args = normalize_arguments(normalized.get("arguments", {}))
    except Exception as exc:
        return ToolCallParseResult(False, raw_text, tool_name=name, error_type="arguments_not_object", error_msg=str(exc), json_repaired=repaired)

    if not isinstance(args, dict):
        return ToolCallParseResult(False, raw_text, tool_name=name, error_type="arguments_not_object", error_msg="arguments must be an object", json_repaired=repaired)
    #校验工具是否在白名单中（如果提供了 known_tools）
    if known_tools is not None and name not in set(known_tools):
        return ToolCallParseResult(False, raw_text, tool_name=name, arguments=args, error_type="unknown_tool", error_msg=f"unknown tool: {name}", json_repaired=repaired)

    return ToolCallParseResult(True, raw_text, tool_name=name, arguments=args, json_repaired=repaired)

#从模型生成的文本中解析出所有工具调用
def parse_tool_calls(text: str, strict: bool = True, known_tools: Optional[Iterable[str]] = None) -> List[ToolCallParseResult]:
    raw_text = clean_terminal_text(text)
    blocks = TOOL_CALL_RE.findall(raw_text)
    if not blocks:
        first = parse_first_tool_call(raw_text, strict=strict, known_tools=known_tools)
        return [] if first.error_type == "no_tool_call" else [first]
    results = []
    for block in blocks:
        results.append(parse_first_tool_call(f"<tool_call>{block}</tool_call>", strict=strict, known_tools=known_tools))
    return results

#判断给定的文本中是否包含「格式错误」的工具调用，即文本中有工具调用的痕迹（标签、花括号等），但无法被严格解析成合法调用。
def has_malformed_tool_call(text: str) -> bool:
    result = parse_first_tool_call(text, strict=True)
    return (not result.ok) and result.error_type not in {"no_tool_call"}

