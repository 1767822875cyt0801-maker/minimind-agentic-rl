#负责 reward 分解和指标计算
import json
import math
import re
from typing import Any, Dict, Iterable, List, Tuple

from agent.tools import safe_eval_expr, validate_tool_args
from agent.trajectory import ToolTrajectory

#基于n-gram重复率，计算文本的重复惩罚值
def rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    toks = re.findall(r"\w+|[^\w\s]", str(text).lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

#将输入转为列表
def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

#从字符串中提取所有数字（整数或浮点数）
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers(text: Any) -> List[float]:
    normalized = str(text).replace(",", "")
    numbers = []
    for match in NUMBER_RE.findall(normalized):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def _is_number_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    return bool(NUMBER_RE.fullmatch(str(value).strip().replace(",", "")))


def _number_answer_match(answer: str, expected: Any, rel_tol: float = 1e-3, abs_tol: float = 1e-2) -> bool:
    try:
        expected_num = float(str(expected).strip().replace(",", ""))
    except Exception:
        return False
    return any(math.isclose(expected_num, n, rel_tol=rel_tol, abs_tol=abs_tol) for n in _numbers(answer))


def _datetime_answer_match(answer: str, expected: Any) -> bool:
    s = str(expected).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
    if not match:
        return False
    year, month, day, hour, minute, second = match.groups()
    expected_parts = [year, str(int(month)), str(int(day)), str(int(hour)), str(int(minute)), str(int(second))]
    answer_parts = {str(int(x)) if x.isdigit() else x for x in re.findall(r"\d+", str(answer).replace(",", ""))}
    required = {year, str(int(hour)), str(int(minute))}
    if not required.issubset(answer_parts):
        return False
    return sum(part in answer_parts for part in expected_parts) >= 4


def _obs_value_used(final_answer: str, value: Any) -> bool:
    if value is None:
        return False
    if _is_number_like(value):
        return _number_answer_match(final_answer, value)
    if _datetime_answer_match(final_answer, value):
        return True
    s = str(value)
    return bool(s and s in final_answer)

#判断模型最终答案是否包含期望答案（或与之数值近似）
def answer_contains_expected(
    answer: str,
    expected: Any,
    tol: float = 1e-6,
    expected_answer_type: str = "",
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-2,
) -> bool:
    if expected is None:
        return True
    if isinstance(expected, list):
        return all(
            answer_contains_expected(
                answer,
                item,
                tol=tol,
                expected_answer_type=expected_answer_type,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for item in expected
        )
    s = str(expected).strip()
    if not s:
        return True
    if expected_answer_type == "number" or _is_number_like(expected):
        return _number_answer_match(answer, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    if s.lower() in str(answer).lower():
        return True
    try:
        expected_num = float(s.replace(",", ""))
    except Exception:
        return False
    return any(abs(expected_num - n) <= tol for n in _numbers(answer))

#轨迹数据提取辅助函数
# 返回所有type == "assistant_tool_call" 的步骤的 meta 字段列表（包含工具名、参数等）
def _tool_steps(traj: ToolTrajectory) -> List[Dict[str, Any]]:
    return [step.meta for step in traj.steps if step.type == "assistant_tool_call"]


def _observation_steps(traj: ToolTrajectory) -> List[Dict[str, Any]]:
    return [step.meta for step in traj.steps if step.type == "tool_observation"]


def _parse_error_count(traj: ToolTrajectory) -> int:
    return sum(1 for step in traj.steps if step.type == "parse_error") + int(traj.metrics.get("malformed_count", 0))


def _normalize_expected_args(task: Dict[str, Any]) -> Dict[str, Any]:
    value = task.get("expected_args") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = {}
    return value if isinstance(value, dict) else {}


def _normalize_expected_obs_keys(task: Dict[str, Any]) -> Dict[str, List[str]]:
    value = task.get("expected_obs_keys") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = {}
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for tool_name, keys in value.items():
        key_list = _as_list(keys)
        normalized[str(tool_name)] = [str(key) for key in key_list if str(key)]
    return normalized


def _args_equivalent(tool_name: str, actual: Dict[str, Any], expected: Dict[str, Any], mode: str) -> bool:
    if not expected:
        return True
    if mode == "none":
        return True
    if mode == "exact":
        return actual == expected
    if tool_name == "calculate_math" and "expression" in expected and "expression" in actual:
        try:
            return math.isclose(float(safe_eval_expr(actual["expression"])), float(safe_eval_expr(expected["expression"])), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            pass
    for key, exp_val in expected.items():
        if key not in actual:
            return False
        act_val = actual[key]
        try:
            if isinstance(exp_val, (int, float)) or isinstance(act_val, (int, float)):
                if not math.isclose(float(act_val), float(exp_val), rel_tol=1e-6, abs_tol=1e-6):
                    return False
            elif str(act_val).replace(" ", "").lower() != str(exp_val).replace(" ", "").lower():
                return False
        except Exception:
            return False
    return True

#核心评估函数
#输入一条完整的ToolTrajectory，返回一个字典，包含评估结果
def _expected_args_for_call(expected_args: Dict[str, Any], tool_name: str, occurrence_idx: int) -> Dict[str, Any]:
    exp = expected_args.get(tool_name, {})
    if isinstance(exp, list):
        return exp[occurrence_idx] if occurrence_idx < len(exp) else {}
    return exp if isinstance(exp, dict) else {}


def evaluate_trajectory(traj: ToolTrajectory, task: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls = _tool_steps(traj)
    observations = _observation_steps(traj)
    called_names = [item.get("name", "") for item in tool_calls]
    expected_sequence = _as_list(task.get("expected_tool_sequence"))
    expected_args = _normalize_expected_args(task)
    expected_obs_keys = _normalize_expected_obs_keys(task)
    args_match_mode = task.get("args_match") or "equivalent"
    final_answer = traj.final_answer or ""
    expected_answer = task.get("expected_answer", task.get("gt"))
    gt = _as_list(task.get("gt"))
    if expected_answer is None and gt:
        expected_answer = gt
    allow_no_tool = bool(task.get("allow_no_tool", False))
    max_tool_calls = task.get("max_tool_calls")

    malformed = _parse_error_count(traj) > 0
    loop = bool(traj.metrics.get("loop_count", 0))
    unfinished = bool(traj.metrics.get("unfinished")) or not traj.done
    overuse = max_tool_calls is not None and len(tool_calls) > int(max_tool_calls)
    has_assistant_final = any(step.type == "assistant_final" for step in traj.steps)
    no_final_after_tool = bool(observations) and not has_assistant_final

    if expected_sequence:
        tool_acc = called_names[: len(expected_sequence)] == expected_sequence and len(called_names) == len(expected_sequence)
    else:
        tool_acc = (len(tool_calls) == 0) if allow_no_tool else (len(tool_calls) > 0 or not expected_answer)

    args_ok = True
    if expected_args:
        occurrence_counts: Dict[str, int] = {}
        for idx, tool_name in enumerate(expected_sequence or called_names):
            matching_call = None
            if idx < len(tool_calls) and tool_calls[idx].get("name") == tool_name:
                matching_call = tool_calls[idx]
            else:
                matching_call = next((call for call in tool_calls if call.get("name") == tool_name), None)
            if matching_call is None:
                args_ok = False
                break
            occurrence_idx = occurrence_counts.get(tool_name, 0)
            occurrence_counts[tool_name] = occurrence_idx + 1
            exp = _expected_args_for_call(expected_args, tool_name, occurrence_idx)
            if not _args_equivalent(tool_name, matching_call.get("arguments") or {}, exp, args_match_mode):
                args_ok = False
                break
    else:
        args_ok = all(validate_tool_args(call.get("name", ""), call.get("arguments") or {}) for call in tool_calls)

    expected_answer_type = task.get("expected_answer_type") or ""
    answer_acc = answer_contains_expected(final_answer, expected_answer, expected_answer_type=expected_answer_type) if expected_answer is not None else bool(final_answer)
    obs_values = []
    default_obs_keys = ["result", "datetime", "characters", "words"]
    for obs in observations:
        result = obs.get("result") if isinstance(obs, dict) else None
        if isinstance(result, dict):
            tool_name = str(obs.get("name", ""))
            keys = expected_obs_keys.get(tool_name) if expected_obs_keys else None
            keys = keys or default_obs_keys
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                if key in result:
                    obs_values.append(str(result[key]))
    obs_use_strict = True if not observations else any(_obs_value_used(final_answer, value) for value in obs_values)
    obs_use = obs_use_strict
    if expected_answer is not None:
        obs_use = obs_use or answer_acc

    if max_tool_calls is not None:
        extra_tool_calls = max(0, len(tool_calls) - int(max_tool_calls))
    elif expected_sequence:
        extra_tool_calls = max(0, len(tool_calls) - len(expected_sequence))
    else:
        extra_tool_calls = 0

    valid = not malformed and not overuse and (allow_no_tool or bool(tool_calls) or expected_sequence == [])

    failures = []
    if not tool_calls and expected_sequence:
        failures.append("no_tool_call")
    if expected_sequence and called_names and not tool_acc:
        failures.append("wrong_tool")
    if malformed:
        failures.append("malformed_json")
    if not args_ok:
        failures.append("wrong_args")
    if observations and not obs_use:
        failures.append("observation_ignored")
    if expected_answer is not None and not answer_acc:
        failures.append("wrong_final_answer")
    if loop:
        failures.append("tool_loop")
    if overuse:
        failures.append("overuse_tool")
    if unfinished:
        failures.append("unfinished_answer")
    if no_final_after_tool:
        failures.append("tool_call_without_final")
    if allow_no_tool and tool_calls:
        failures.append("unnecessary_tool")

    return {
        "valid": valid,
        "tool_acc": tool_acc,
        "args_acc": args_ok,
        "obs_use_acc": obs_use,
        "obs_use_strict_acc": obs_use_strict,
        "answer_acc": answer_acc,
        "loop": loop,
        "malformed": malformed,
        "unfinished": unfinished,
        "tool_call_without_final": no_final_after_tool,
        "tool_calls": len(tool_calls),
        "extra_tool_calls": extra_tool_calls,
        "failures": failures,
        "called_tools": called_names,
        "final_answer": final_answer,
    }

#总奖励计算
def _expected_tool_count(task: Dict[str, Any], metrics: Dict[str, Any]) -> int:
    expected_sequence = _as_list(task.get("expected_tool_sequence"))
    if expected_sequence:
        return len(expected_sequence)
    if task.get("max_tool_calls") is not None:
        return int(task.get("max_tool_calls") or 0)
    return int(metrics.get("tool_calls", 0))


def _concise_final_reward(final_answer: str) -> float:
    length = len(str(final_answer).strip())
    if 1 <= length <= 80:
        return 0.15
    if length <= 160:
        return 0.05
    return -0.10


def _compute_default_reward(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format_reward": 0.3 if not metrics["malformed"] else -0.7,
        "tool_name_reward": 0.7 if metrics["tool_acc"] else -0.5,
        "args_reward": 0.6 if metrics["args_acc"] else -0.5,
        "obs_use_reward": 0.6 if metrics["obs_use_acc"] else -0.5,
        "final_answer_reward": 1.0 if metrics["answer_acc"] else -0.8,
        "loop_penalty": -0.6 if metrics["loop"] else 0.0,
        "overuse_penalty": -0.4 if "overuse_tool" in metrics["failures"] else 0.0,
        "unnecessary_tool_penalty": -0.5 if "unnecessary_tool" in metrics["failures"] else 0.0,
        "unfinished_penalty": -0.5 if metrics["unfinished"] else 0.0,
        "repetition_penalty": -rep_penalty(metrics.get("final_answer", "")),
    }


def _compute_dense_v2_reward(metrics: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    expected_calls = _expected_tool_count(task, metrics)
    tool_calls = int(metrics.get("tool_calls", 0))
    extra_calls = max(int(metrics.get("extra_tool_calls", 0)), max(0, tool_calls - expected_calls))
    step_efficiency = 0.25 if tool_calls == expected_calls else max(-0.25, -0.10 * abs(tool_calls - expected_calls))
    return {
        "format_reward": 0.20 if not metrics["malformed"] else -0.70,
        "tool_name_reward": 0.35 if metrics["tool_acc"] else -0.70,
        "args_reward": 0.40 if metrics["args_acc"] else -0.60,
        "obs_use_reward": 0.35 if metrics.get("obs_use_strict_acc", metrics["obs_use_acc"]) else -0.35,
        "final_answer_reward": 1.10 if metrics["answer_acc"] else -1.00,
        "step_efficiency_reward": step_efficiency,
        "concise_final_reward": _concise_final_reward(metrics.get("final_answer", "")),
        "extra_tool_penalty": -0.20 * extra_calls,
        "loop_penalty": -0.60 if metrics["loop"] else 0.0,
        "overuse_penalty": -0.40 if "overuse_tool" in metrics["failures"] else 0.0,
        "unnecessary_tool_penalty": -0.60 if "unnecessary_tool" in metrics["failures"] else 0.0,
        "unfinished_penalty": -0.60 if metrics["unfinished"] else 0.0,
        "repetition_penalty": -rep_penalty(metrics.get("final_answer", "")),
    }


def compute_total_reward(traj: ToolTrajectory, task: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    metrics = evaluate_trajectory(traj, task)
    profile = task.get("reward_profile") or "default"
    reward_info = _compute_dense_v2_reward(metrics, task) if profile == "dense_v2" else _compute_default_reward(metrics)
    reward_info["reward_profile"] = profile
    reward = sum(value for value in reward_info.values() if isinstance(value, (int, float, bool)))
    reward = max(min(reward, 3.0), -3.0)
    reward_info.update(metrics)
    return reward, reward_info


def aggregate_reward_infos(reward_infos: List[Dict[str, Any]]) -> Dict[str, float]:
    if not reward_infos:
        return {}
    numeric_keys = [
        "format_reward",
        "tool_name_reward",
        "args_reward",
        "obs_use_reward",
        "final_answer_reward",
        "step_efficiency_reward",
        "concise_final_reward",
        "extra_tool_penalty",
        "loop_penalty",
        "overuse_penalty",
        "unnecessary_tool_penalty",
        "unfinished_penalty",
        "repetition_penalty",
        "valid",
        "tool_acc",
        "args_acc",
        "obs_use_acc",
        "obs_use_strict_acc",
        "answer_acc",
        "loop",
        "malformed",
        "unfinished",
        "tool_calls",
        "extra_tool_calls",
    ]
    summary = {}
    for key in numeric_keys:
        values = []
        for info in reward_infos:
            value = info.get(key)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            summary[key] = sum(values) / len(values)
    return summary
