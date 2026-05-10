import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.reward import compute_total_reward
from agent.tool_env import ToolUseEnv
from agent.trajectory import trajectory_to_dict


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def init_local_model(args):
    import torch
    from transformers import AutoTokenizer

    from models.model_minimind import MiniMindConfig, MiniMindForCausalLM

    load_from = Path(args.load_from)
    if not load_from.is_absolute():
        load_from = PROJECT_ROOT / load_from
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir

    tokenizer = AutoTokenizer.from_pretrained(str(load_from))
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config)
    moe_suffix = "_moe" if args.use_moe else ""
    ckp = save_dir / f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    model.load_state_dict(torch.load(str(ckp), map_location=args.device), strict=True)
    model = model.eval().to(args.device)
    if "cuda" in str(args.device):
        model = model.half()
    return model, tokenizer


def generate_local(model, tokenizer, env: ToolUseEnv, args) -> str:
    import torch

    input_text = tokenizer.apply_chat_template(
        env.get_messages(),
        tokenize=False,
        add_generation_prompt=True,
        tools=env.get_tool_schemas() or None,
        open_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)

#评测 rollout 的主入口
#创建环境、reset、循环生成和 env.step()，最后拿 trajectory 调用 compute_total_reward() 并返回结果
def run_eval_case(item: Dict[str, Any], model, tokenizer, args) -> Dict[str, Any]:
    #创建 ToolUseEnv（模式、最大工具调用次数）
    env = ToolUseEnv(mode=args.mode, max_tool_calls=int(item.get("max_tool_calls", args.max_tool_calls)))
    #初始化环境（设置任务、工具、初始消息）
    env.reset(item)
    for _ in range(args.max_turns):
        model_output = generate_local(model, tokenizer, env, args)
        info = env.step(model_output)
        if info.get("done"):
            break
    #若循环结束环境未完成
    if not env.done:
        env.mark_unfinished()
    #获取轨迹 ToolTrajectory
    traj = env.get_trajectory()
    reward, info = compute_total_reward(traj, item)
    traj.reward = reward
    traj.metrics.update(info)
    return {"item": item, "reward": reward, "metrics": info, "trajectory": trajectory_to_dict(traj)}


def build_error_result(item: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    exception_type = type(error).__name__
    exception_msg = str(error)[:500]
    metrics = {
        "valid": False,
        "tool_acc": False,
        "args_acc": False,
        "obs_use_acc": False,
        "answer_acc": False,
        "loop": False,
        "malformed": False,
        "unfinished": True,
        "tool_call_without_final": False,
        "tool_calls": 0,
        "failures": ["eval_exception"],
        "called_tools": [],
        "final_answer": "",
        "exception_type": exception_type,
        "exception_msg": exception_msg,
    }
    trajectory = {
        "task_id": str(item.get("id") or item.get("task_id") or ""),
        "prompt": item.get("prompt", ""),
        "tools": item.get("tools", []),
        "steps": [],
        "final_answer": "",
        "done": False,
        "reward": -3.0,
        "metrics": metrics,
    }
    return {"item": item, "reward": -3.0, "metrics": metrics, "trajectory": trajectory}


def run_eval_items(items: List[Dict[str, Any]], model, tokenizer, args) -> List[Dict[str, Any]]:
    results = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.get('id', '')}: {item.get('prompt', '')[:80]}")
        try:
            results.append(run_eval_case(item, model, tokenizer, args))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[eval_exception] {item.get('id', '')}: {type(exc).__name__}: {str(exc)[:300]}")
            results.append(build_error_result(item, exc))
    return results


#计算某个布尔指标（如 "valid"）在所有结果中的平均值（比例）
def average_bool(results: Iterable[Dict[str, Any]], key: str) -> float:
    values = [1.0 if result["metrics"].get(key) else 0.0 for result in results]
    return sum(values) / max(len(values), 1)

#汇总所有样本的奖励、指标、轨迹等信息
def build_summary(results: List[Dict[str, Any]], args) -> Dict[str, Any]:
    return {
        "model": args.weight,
        "mode": args.mode,
        "num_cases": len(results),
        #有效轨迹比例。模型这一轮回答是否形成了一条基本可解析、可执行、可结束的有效 trajectory
        "valid_rate": average_bool(results, "valid"),
        #工具选择准确率
        "tool_acc": average_bool(results, "tool_acc"),
        #工具参数准确率
        "args_acc": average_bool(results, "args_acc"),
        #工具观察结果使用准确率。模型调用工具之后，是否真的使用了工具返回的 observation
        "obs_use_acc": average_bool(results, "obs_use_acc"),
        #最终答案准确率。最终自然语言回答是否包含或等价于 expected answer
        #最综合的指标，几乎受前面所有指标影响
        "answer_acc": average_bool(results, "answer_acc"),
        #工具调用循环率。模型是否重复调用同一个工具，或者进入工具调用循环
        "loop_rate": average_bool(results, "loop"),
        #格式错误率。模型生成的 tool_call 格式不合法，导致 parser 解析失败。模型生成的 tool_call 格式不合法，导致 parser 解析失败
        "malformed_rate": average_bool(results, "malformed"),
        #未完成轨迹比例。模型这一轮回答是否形成一条未完成的 trajectory
        "unfinished_rate": average_bool(results, "unfinished"),
        #平均工具调用次数
        "avg_tool_calls": sum(r["metrics"].get("tool_calls", 0) for r in results) / max(len(results), 1),
        #平均总奖励
        "avg_reward": sum(r["reward"] for r in results) / max(len(results), 1),
    }


def write_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any], args) -> None:
    output = Path(args.output) if args.output else PROJECT_ROOT / "evals" / "reports" / f"{args.weight}_{args.mode}.json"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    failures_path = output.with_name(output.stem + "_failures.jsonl")
    traj_path = output.with_name(output.stem + "_trajectories.jsonl")
    with failures_path.open("w", encoding="utf-8", newline="\n") as f_fail, traj_path.open("w", encoding="utf-8", newline="\n") as f_traj:
        for result in results:
            f_traj.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["metrics"].get("failures"):
                f_fail.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {output}")
    print(f"failures: {failures_path}")
    print(f"trajectories: {traj_path}")


def main():
    parser = argparse.ArgumentParser(description="MiniMind Tool-Use Agent evaluation")
    parser.add_argument("--eval_path", default="evals/tool_eval.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", default="raw", choices=["raw", "guarded"])
    parser.add_argument("--weight", default="full_sft")

    #分词器/配置目录
    parser.add_argument("--load_from", default="models")
    parser.add_argument("--save_dir", default="out")

    #模型结构参数（hidden_size, num_hidden_layers, use_moe）
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_turns", default=4, type=int)

    #每个任务最大工具调用次数（可被任务覆盖）
    parser.add_argument("--max_tool_calls", default=3, type=int)

    #生成参数（max_new_tokens, temperature, top_p, do_sample）
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--do_sample", action="store_true")

    #限制评估样本数量（调试用）
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--continue_on_error", action="store_true", help="单条样本异常时记录 eval_exception 并继续评测")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = PROJECT_ROOT / eval_path
    items = read_jsonl(eval_path)
    if args.limit:
        items = items[: args.limit]

    model, tokenizer = init_local_model(args)
    start = time.time()
    results = run_eval_items(items, model, tokenizer, args)
    summary = build_summary(results, args)
    summary["elapsed_sec"] = round(time.time() - start, 3)
    write_outputs(results, summary, args)


if __name__ == "__main__":
    main()
