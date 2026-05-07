#环境状态机，实现了 Agent 与工具交互的模拟环境
import copy
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.parser import ToolCallParseResult, clean_terminal_text, parse_first_tool_call
from agent.tools import execute_tool_call, get_tool_name, get_tool_names, get_tool_schemas
from agent.trajectory import ToolTrajectory, TrajectoryStep


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _scalar_to_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_tool_answer(name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result)
    if "error" in result:
        return f"工具执行失败：{result['error']}"
    if name == "calculate_math":
        expression = result.get("expression") or arguments.get("expression", "")
        value = _scalar_to_text(result.get("result"))
        return f"{expression} = {value}" if expression else value
    if name == "get_current_time":
        dt = result.get("datetime", "")
        timezone = result.get("timezone", "")
        return f"当前时间是 {dt}（{timezone}）" if timezone else f"当前时间是 {dt}"
    if name == "random_number":
        return f"生成的随机数是 {_scalar_to_text(result.get('result'))}"
    if name == "text_length":
        return f"这段文本共有 {result.get('characters')} 个字符，{result.get('words')} 个单词"
    if name == "unit_converter":
        return (
            f"{_scalar_to_text(result.get('value'))} {result.get('from_unit', '')} = "
            f"{_scalar_to_text(result.get('result'))} {result.get('to_unit', '')}"
        )
    return json.dumps(result, ensure_ascii=False)


def clean_final_text(text: str) -> str:
    text = clean_terminal_text(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL).strip()
    if "<tool_call>" in text or "</tool_call>" in text:
        return ""
    return text


class ToolUseEnv:
    def __init__(self, mode: str = "raw", max_tool_calls: int = 3, allow_guardrails: Optional[bool] = None):
        if mode not in {"raw", "guarded"}:
            raise ValueError("mode must be raw or guarded")
        self.mode = mode
        self.max_tool_calls = max_tool_calls
        self.allow_guardrails = (mode == "guarded") if allow_guardrails is None else allow_guardrails
        self.reset({})

    def reset(self, task: Dict[str, Any]):
        #存储当前任务原始字典（如 prompt、tools、期望答案等）
        self.task = task or {}
        self.task_id = str(self.task.get("id") or self.task.get("task_id") or "")
        self.prompt = self.task.get("prompt") or self._prompt_from_messages(self.task.get("messages", []))
        self.tools = self._resolve_tools(self.task.get("tools"))
        self.tool_names = get_tool_names(self.tools)
        self.messages = self._initial_messages(self.task)
        self.tool_history: List[Dict[str, Any]] = []
        self.executed_signatures = set()
        self.done = False
        self.final_answer: Optional[str] = None
        self.metrics = {
            "tool_call_count": 0,
            "parse_error_count": 0,
            "malformed_count": 0,
            "loop_count": 0,
            "guarded_stop_count": 0,
            "json_repaired_count": 0,
            "unfinished": False,
        }
        self.trajectory = ToolTrajectory(
            task_id=self.task_id,
            prompt=self.prompt,
            tools=list(self.tool_names),
            metrics=self.metrics,
        )
        if self.prompt:
            self.trajectory.steps.append(TrajectoryStep("user", "user", self.prompt))
        return self

    def _resolve_tools(self, tools: Any) -> List[Dict[str, Any]]:
        if not tools:
            return []
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                tools = [tools]
        if isinstance(tools, list):
            if all(isinstance(item, str) for item in tools):
                return get_tool_schemas(tools)
            return get_tool_schemas(tools)
        return []

    def _initial_messages(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        if task.get("messages"):
            messages = []
            for msg in copy.deepcopy(task["messages"]):
                if isinstance(msg, dict):
                    msg.pop("tools", None)
                    messages.append(msg)
            return messages
        if self.prompt:
            return [{"role": "user", "content": self.prompt}]
        return []

    @staticmethod
    def _prompt_from_messages(messages: Iterable[Dict[str, Any]]) -> str:
        for msg in messages or []:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def get_messages(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.messages)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.tools)

    def get_trajectory(self) -> ToolTrajectory:
        self.trajectory.done = self.done
        self.trajectory.final_answer = self.final_answer
        self.trajectory.metrics = dict(self.metrics)
        return self.trajectory

    def mark_unfinished(self) -> None:
        self.metrics["unfinished"] = True
        self.trajectory.metrics = dict(self.metrics)

    def _signature(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, str]:
        return name, _json_dumps(arguments)

    def step(self, model_output: str) -> Dict[str, Any]:
        if self.done:
            return {"type": "already_done", "done": True}
        ## 1. 清理文本
        output = clean_terminal_text(model_output)
        # 2. 解析第一个工具调用
        parse = parse_first_tool_call(output, strict=(self.mode == "raw"), known_tools=self.tool_names or None)
        # 3. 统计 JSON 修复次数
        if parse.json_repaired:
            self.metrics["json_repaired_count"] += 1
        # 4. 根据解析结果分发
        if parse.ok:
            return self._handle_tool_call(output, parse)
        return self._handle_non_tool_output(output, parse)
    #模型输出有效工具调用时
    def _handle_tool_call(self, output: str, parse: ToolCallParseResult) -> Dict[str, Any]:
        name = parse.tool_name or ""
        arguments = parse.arguments or {}
        signature = self._signature(name, arguments)
        is_loop = signature in self.executed_signatures

        self.messages.append({"role": "assistant", "content": output})
        self.trajectory.steps.append(
            TrajectoryStep(
                role="assistant",
                type="assistant_tool_call",
                content=output,
                meta={"name": name, "arguments": arguments, "json_repaired": parse.json_repaired, "loop": is_loop},
            )
        )

        if is_loop:
            self.metrics["loop_count"] += 1
            if self.allow_guardrails:
                self.metrics["guarded_stop_count"] += 1
                self.done = True
                self.final_answer = self.render_final_answer()
                self.trajectory.steps.append(
                    TrajectoryStep("system", "runtime_guard", "repeated tool call blocked", {"name": name, "arguments": arguments})
                )
                return {"type": "runtime_guard", "done": True, "loop": True, "parse": parse}

        if self.metrics["tool_call_count"] >= self.max_tool_calls:
            self.metrics["guarded_stop_count"] += 1
            self.done = True
            self.final_answer = self.render_final_answer()
            self.trajectory.steps.append(TrajectoryStep("system", "runtime_guard", "max tool calls reached"))
            return {"type": "runtime_guard", "done": True, "parse": parse}

        self.metrics["tool_call_count"] += 1
        self.executed_signatures.add(signature)
        result = execute_tool_call(name, arguments)
        result_text = json.dumps(result, ensure_ascii=False)
        self.tool_history.append({"name": name, "arguments": arguments, "result": result})
        self.messages.append({"role": "tool", "content": result_text})
        self.trajectory.steps.append(
            TrajectoryStep("tool", "tool_observation", result_text, {"name": name, "arguments": arguments, "result": result})
        )
        return {"type": "tool_observation", "observation": result, "done": False, "parse": parse, "loop": is_loop}

    #模型输出没有有效工具调用时
    def _handle_non_tool_output(self, output: str, parse: ToolCallParseResult) -> Dict[str, Any]:
        if parse.error_type != "no_tool_call":
            self.metrics["parse_error_count"] += 1
            self.metrics["malformed_count"] += 1
            self.trajectory.steps.append(
                TrajectoryStep(
                    role="assistant",
                    type="parse_error",
                    content=output,
                    meta={"error_type": parse.error_type, "error_msg": parse.error_msg},
                )
            )
            if self.mode == "raw":
                self.done = True
                self.final_answer = ""
                return {"type": "parse_error", "done": True, "parse": parse}

        self.messages.append({"role": "assistant", "content": output})
        self.final_answer = clean_final_text(output) or (self.render_final_answer() if self.allow_guardrails else output)
        self.done = True
        self.trajectory.steps.append(TrajectoryStep("assistant", "assistant_final", self.final_answer, {"raw": output}))
        return {"type": "final_answer", "done": True, "answer": self.final_answer, "parse": parse}
    # 工具调用结果渲染。当模型没有给出明确 final answer 但环境需要结束时（如 guard 阻止、达到最大调用次数），根据已执行的工具历史生成一个回答
    def render_final_answer(self) -> str:
        if self.final_answer:
            return self.final_answer
        if not self.tool_history:
            return ""
        if len(self.tool_history) == 1:
            item = self.tool_history[0]
            return format_tool_answer(item["name"], item["arguments"], item["result"])
        return "；".join(format_tool_answer(item["name"], item["arguments"], item["result"]) for item in self.tool_history)

