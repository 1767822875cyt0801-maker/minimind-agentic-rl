import argparse
import json
import os
from pathlib import Path


CALCULATE_MATH_TOOL = {
    "function": {
        "name": "calculate_math",
        "description": "计算数学表达式的结果，支持加减乘除、幂运算、平方、平方根、开方和sqrt函数",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如12**2、sqrt(144)、math.sqrt(144)"
                }
            },
            "required": ["expression"]
        }
    }
}


def build_sample(prompt, gt):
    return {
        "conversations": [
            {
                "role": "system",
                "content": "",
                "tools": json.dumps([CALCULATE_MATH_TOOL], ensure_ascii=False)
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        "gt": [str(x) for x in gt]
    }


def build_samples():
    numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 25, 30]
    samples = []

    for n in numbers:
        square = n * n
        samples.extend([
            build_sample(f"请计算 {n} 的平方", [square]),
            build_sample(f"请计算 {square} 的平方根", [n]),
            build_sample(f"把 {square} 开方是多少？", [n]),
            build_sample(f"根号 {square} 等于多少？", [n]),
            build_sample(f"What is the square of {n}?", [square]),
            build_sample(f"What is the square root of {square}?", [n]),
        ])

    contrast_numbers = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    for n in contrast_numbers:
        square = n * n
        samples.extend([
            build_sample(f"先算 {n} 的平方，再算 {square} 的平方根", [square, n]),
            build_sample(f"不要把平方根算成平方：{square} 的平方根是多少？", [n]),
            build_sample(f"请区分平方和平方根：{n} 的平方是多少，{square} 的平方根是多少？", [square, n]),
            build_sample(f"Compare square and square root: square of {n}, then square root of {square}.", [square, n]),
        ])

    return samples


def build_sft_sample(prompt, expressions):
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "calculate_math",
                "arguments": {
                    "expression": expression
                }
            }
        }
        for expression in expressions
    ]

    return {
        "conversations": [
            {
                "role": "system",
                "content": (
                    "你可以调用 calculate_math 工具计算数学表达式。"
                    "遇到平方使用 n**2；遇到平方根、开方、根号或 square root 使用 sqrt(n)。"
                    "需要计算时只输出工具调用。"
                )
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": json.dumps(tool_calls, ensure_ascii=False)
            }
        ]
    }


def build_sft_samples():
    numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 25, 30]
    samples = []

    for n in numbers:
        square = n * n
        samples.extend([
            build_sft_sample(f"请计算 {n} 的平方", [f"{n}**2"]),
            build_sft_sample(f"请计算 {square} 的平方根", [f"sqrt({square})"]),
            build_sft_sample(f"把 {square} 开方是多少？", [f"sqrt({square})"]),
            build_sft_sample(f"根号 {square} 等于多少？", [f"sqrt({square})"]),
            build_sft_sample(f"What is the square of {n}?", [f"{n}**2"]),
            build_sft_sample(f"What is the square root of {square}?", [f"sqrt({square})"]),
        ])

    contrast_numbers = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    for n in contrast_numbers:
        square = n * n
        samples.extend([
            build_sft_sample(f"先算 {n} 的平方，再算 {square} 的平方根", [f"{n}**2", f"sqrt({square})"]),
            build_sft_sample(f"不要把平方根算成平方：{square} 的平方根是多少？", [f"sqrt({square})"]),
            build_sft_sample(f"请区分平方和平方根：{n} 的平方是多少，{square} 的平方根是多少？", [f"{n}**2", f"sqrt({square})"]),
            build_sft_sample(f"Compare square and square root: square of {n}, then square root of {square}.", [f"{n}**2", f"sqrt({square})"]),
        ])

    return samples


def read_existing_prompts(path):
    prompts = set()
    if not path.exists():
        return prompts

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            for conv in sample.get("conversations", []):
                if conv.get("role") == "user":
                    prompts.add(conv.get("content", ""))
                    break
    return prompts


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def append_missing(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_prompts = read_existing_prompts(path)
    missing = []

    for sample in samples:
        prompt = next(
            conv.get("content", "")
            for conv in sample["conversations"]
            if conv.get("role") == "user"
        )
        if prompt not in existing_prompts:
            missing.append(sample)
            existing_prompts.add(prompt)

    if missing:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            for sample in missing:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(missing)


def main():
    parser = argparse.ArgumentParser(description="生成平方/平方根对比 tool-call RL 数据")
    parser.add_argument("--addon_path", default="../dataset/agent_rl_math_sqrt_square.jsonl", help="单独保存新增样本的路径")
    parser.add_argument("--sft_path", default="../dataset/sft_toolcall_sqrt_square.jsonl", help="单独保存SFT工具调用标注样本的路径")
    parser.add_argument("--append_path", default="../dataset/agent_rl_math.jsonl", help="追加到现有数学RL数据集的路径")
    parser.add_argument("--no_append", action="store_true", help="只生成addon文件，不追加到现有数据集")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    addon_path = (script_dir / args.addon_path).resolve()
    sft_path = (script_dir / args.sft_path).resolve()
    append_path = (script_dir / args.append_path).resolve()

    samples = build_samples()
    sft_samples = build_sft_samples()
    write_jsonl(addon_path, samples)
    write_jsonl(sft_path, sft_samples)

    appended = 0
    if not args.no_append:
        appended = append_missing(append_path, samples)

    print(f"addon samples: {len(samples)} -> {addon_path}")
    print(f"sft samples: {len(sft_samples)} -> {sft_path}")
    if not args.no_append:
        print(f"appended samples: {appended} -> {append_path}")


if __name__ == "__main__":
    main()
