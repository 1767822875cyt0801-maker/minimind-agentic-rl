import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def short(text: Any, limit: int = 160) -> str:
    value = str(text or "").replace("\n", " ")
    return value[:limit] + ("..." if len(value) > limit else "")


def task_prompt(record: Dict[str, Any]) -> str:
    traj = record.get("trajectory") or {}
    return str(traj.get("prompt") or record.get("prompt") or "")


def summarize_trace(records: List[Dict[str, Any]], zero_threshold: float = 1e-6) -> Dict[str, Any]:
    rewards = [float(record.get("reward", 0.0)) for record in records]
    failures = Counter()
    for record in records:
        failures.update(record.get("failures") or [])

    by_step = defaultdict(list)
    for record in records:
        by_step[int(record.get("step", 0))].append(record)

    group_stats = []
    for step, group in sorted(by_step.items()):
        values = [float(item.get("reward", 0.0)) for item in group]
        std = pstdev(values) if len(values) > 1 else 0.0
        group_failures = Counter()
        for item in group:
            group_failures.update(item.get("failures") or [])
        group_stats.append(
            {
                "step": step,
                "count": len(group),
                "reward_min": min(values) if values else 0.0,
                "reward_max": max(values) if values else 0.0,
                "reward_mean": mean(values) if values else 0.0,
                "reward_std": std,
                "zero_group": std < zero_threshold,
                "failures": group_failures.most_common(),
                "task_id": str(group[0].get("task_id", "")) if group else "",
                "prompt": short(task_prompt(group[0])) if group else "",
            }
        )

    nonzero_groups = [item for item in group_stats if not item["zero_group"]]
    bad_examples = sorted(
        records,
        key=lambda item: (float(item.get("reward", 0.0)), len(item.get("failures") or [])),
    )[:10]

    return {
        "rows": len(records),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_dist": Counter(round(value, 4) for value in rewards).most_common(30),
        "failure_counts": failures.most_common(),
        "zero_group_steps": sum(1 for item in group_stats if item["zero_group"]),
        "total_steps": len(group_stats),
        "zero_group_rate": sum(1 for item in group_stats if item["zero_group"]) / max(len(group_stats), 1),
        "nonzero_group_examples": nonzero_groups[:20],
        "worst_examples": [
            {
                "step": item.get("step"),
                "task_id": item.get("task_id"),
                "reward": item.get("reward"),
                "failures": item.get("failures") or [],
                "called_tools": item.get("called_tools") or [],
                "prompt": short(task_prompt(item)),
                "final_answer": short(item.get("final_answer")),
            }
            for item in bad_examples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Agent RL rollout trace JSONL")
    parser.add_argument("trace_path")
    parser.add_argument("--output", default="")
    parser.add_argument("--zero_threshold", type=float, default=1e-6)
    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    records = read_jsonl(trace_path)
    summary = summarize_trace(records, zero_threshold=args.zero_threshold)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
