import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def raw_guarded_gap(raw_path: Path, guarded_path: Path) -> Dict:
    raw = read_json(raw_path)
    guarded = read_json(guarded_path)
    keys = [
        "valid_rate",
        "tool_acc",
        "args_acc",
        "obs_use_acc",
        "answer_acc",
        "loop_rate",
        "malformed_rate",
        "unfinished_rate",
    ]
    return {
        "ablation": "raw_guarded_gap",
        "raw": str(raw_path),
        "guarded": str(guarded_path),
        "metrics": {
            key: {
                "raw": raw.get(key),
                "guarded": guarded.get(key),
                "gap": None if raw.get(key) is None or guarded.get(key) is None else guarded.get(key) - raw.get(key),
            }
            for key in keys
        },
        "interpretation": "A large positive guarded gap means runtime guardrails are carrying more of the system result.",
    }


def reward_components(paths: Iterable[Path]) -> Dict:
    sums = defaultdict(float)
    counts = defaultdict(int)
    total_rows = 0
    for path in paths:
        for row in read_jsonl(path):
            total_rows += 1
            metrics = row.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, bool):
                    sums[key] += float(value)
                    counts[key] += 1
                elif isinstance(value, (int, float)):
                    sums[key] += float(value)
                    counts[key] += 1
    return {
        "ablation": "reward_components",
        "num_trajectories": total_rows,
        "means": {key: sums[key] / counts[key] for key in sorted(sums) if counts[key]},
    }


def failure_counts(paths: Iterable[Path]) -> Dict:
    counter = Counter()
    rows = 0
    for path in paths:
        for row in read_jsonl(path):
            rows += 1
            for failure in row.get("metrics", {}).get("failures", []):
                counter[failure] += 1
    return {"ablation": "failure_counts", "num_failed_rows": rows, "failures": dict(counter.most_common())}


def main():
    parser = argparse.ArgumentParser(description="Summarize Agentic RL ablation reports")
    parser.add_argument("--ablation", required=True, choices=["raw_guarded_gap", "reward_components", "failure_counts"])
    parser.add_argument("--raw", default="evals/reports/full_sft_raw.json")
    parser.add_argument("--guarded", default="evals/reports/full_sft_guarded.json")
    parser.add_argument("--reports", nargs="*", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.ablation == "raw_guarded_gap":
        result = raw_guarded_gap(Path(args.raw), Path(args.guarded))
    elif args.ablation == "reward_components":
        result = reward_components([Path(p) for p in args.reports])
    else:
        result = failure_counts([Path(p) for p in args.reports])

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

