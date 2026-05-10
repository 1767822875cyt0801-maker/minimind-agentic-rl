import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools import execute_tool_call, get_tool_schemas


DEFAULT_EVAL_SIZE = 1000
DEFAULT_SFT_SEED_SIZE = 12000
DEFAULT_SEED = 42
CURRENT_TOOL_NAMES = ["calculate_math", "text_length", "unit_converter", "get_current_time"]
PROMPT_MODES = ("original", "normalized", "both")

EXPR_SPAN_RE = re.compile(r"(?:math\.)?sqrt\s*\([^)]*\)|[0-9+\-*/().%^ \t]+", re.IGNORECASE)


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_call(name: str, arguments: Dict) -> List[Dict]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def normalize_prompt_text(text: str) -> str:
    replacements = {
        "（": "(",
        "）": ")",
        "×": "*",
        "x": "*",
        "X": "*",
        "÷": "/",
        "－": "-",
        "—": "-",
        "^": "**",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_expression(expr: str) -> str:
    expr = normalize_prompt_text(expr)
    expr = re.sub(r"\s+", "", expr)
    expr = expr.strip(".,，。?？!！;；:：")
    while expr and expr[-1] in "+-*/%^(":
        expr = expr[:-1]
    while expr and expr[0] in "*/%).,，。?？!！;；:：":
        expr = expr[1:]
    return expr


def looks_like_math_expression(expr: str) -> bool:
    if not expr:
        return False
    if "sqrt(" in expr.lower():
        return True
    return bool(re.search(r"\d\s*(?:\*\*|[+\-*/%^])\s*\d", expr))


def extract_math_expressions(text: str) -> List[str]:
    text = normalize_prompt_text(text)
    expressions: List[str] = []
    seen = set()
    for match in EXPR_SPAN_RE.finditer(text):
        expr = clean_expression(match.group(0))
        if not looks_like_math_expression(expr):
            continue
        result = execute_tool_call("calculate_math", {"expression": expr})
        if "error" in result:
            continue
        normalized = result.get("expression", expr)
        if normalized not in seen:
            expressions.append(normalized)
            seen.add(normalized)
    return expressions


def numeric_match(actual, expected, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    try:
        return math.isclose(float(actual), float(str(expected).replace(",", "")), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def last_user_prompt(conversations: Sequence[Dict]) -> str:
    for message in reversed(conversations):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def final_assistant_is_empty(conversations: Sequence[Dict]) -> bool:
    if not conversations:
        return False
    final = conversations[-1]
    return final.get("role") == "assistant" and not str(final.get("content") or "").strip()


def verify_expressions(expressions: Sequence[str], gt: Sequence[str]) -> Optional[List[Dict]]:
    if not gt or len(expressions) != len(gt):
        return None
    steps = []
    for expr, expected in zip(expressions, gt):
        result = execute_tool_call("calculate_math", {"expression": expr})
        if "error" in result or not numeric_match(result.get("result"), expected):
            return None
        steps.append({"expression": result["expression"], "result": result["result"]})
    return steps


def normalized_prompt(expressions: Sequence[str]) -> str:
    if len(expressions) == 1:
        return f"Use the calculator tool to compute this expression: {expressions[0]}."
    joined = "; ".join(expressions)
    return (
        "Call the calculator tool once for each expression, in order, "
        f"then give the results in the same order: {joined}."
    )


def build_task_row(
    source: str,
    row_index: int,
    prompt: str,
    original_prompt: str,
    steps: Sequence[Dict],
    gt: Sequence[str],
    prompt_mode: str,
) -> Dict:
    category = "converted_math_single" if len(steps) == 1 else "converted_math_multi"
    expected_args = {"calculate_math": {"expression": steps[0]["expression"]}}
    if len(steps) > 1:
        expected_args = {"calculate_math": [{"expression": step["expression"]} for step in steps]}

    expected_answer = gt[0] if len(gt) == 1 else list(gt)
    suffix = "" if prompt_mode == "original" else f"_{prompt_mode}"
    return {
        "id": f"converted_{source}_{row_index:06d}{suffix}",
        "category": category,
        "source": source,
        "prompt_mode": prompt_mode,
        "original_prompt": original_prompt,
        "prompt": prompt,
        "tools": CURRENT_TOOL_NAMES,
        "expected_tool_sequence": ["calculate_math"] * len(steps),
        "expected_args": expected_args,
        "expected_obs_keys": {"calculate_math": ["result"]},
        "expected_answer": expected_answer,
        "expected_answer_type": "number",
        "gt": list(gt),
        "max_tool_calls": len(steps),
        "reward_profile": "dense_v2",
    }


def convert_row(
    row: Dict,
    source: str,
    row_index: int,
    prompt_mode: str = "original",
    max_expressions: int = 3,
) -> Optional[Dict]:
    rows = convert_row_variants(
        row,
        source,
        row_index,
        prompt_mode=prompt_mode,
        include_single=True,
        include_multi=True,
        max_expressions=max_expressions,
    )
    return rows[0] if rows else None


def convert_row_variants(
    row: Dict,
    source: str,
    row_index: int,
    prompt_mode: str = "original",
    include_single: bool = True,
    include_multi: bool = True,
    max_expressions: int = 3,
) -> List[Dict]:
    conversations = row.get("conversations") or []
    gt = [str(item) for item in row.get("gt") or [] if str(item).strip()]
    if not final_assistant_is_empty(conversations) or not gt:
        return []

    prompt = last_user_prompt(conversations)
    expressions = extract_math_expressions(prompt)
    steps = verify_expressions(expressions, gt)
    if not steps:
        return []
    if max_expressions and len(steps) > max_expressions:
        return []
    if len(steps) == 1 and not include_single:
        return []
    if len(steps) > 1 and not include_multi:
        return []

    modes = ["original", "normalized"] if prompt_mode == "both" else [prompt_mode]
    rows = []
    for mode in modes:
        if mode not in {"original", "normalized"}:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode}")
        row_prompt = prompt if mode == "original" else normalized_prompt([step["expression"] for step in steps])
        rows.append(build_task_row(source, row_index, row_prompt, prompt, steps, gt, mode))
    return rows


def sft_sample_from_task(row: Dict) -> Dict:
    conversations = [
        {"role": "system", "content": "", "tools": json.dumps(get_tool_schemas(row["tools"]), ensure_ascii=False)},
        {"role": "user", "content": row["prompt"]},
    ]
    expected_args = row["expected_args"]["calculate_math"]
    args_list = expected_args if isinstance(expected_args, list) else [expected_args]
    results = []
    for arguments in args_list:
        result = execute_tool_call("calculate_math", arguments)
        results.append(str(row["gt"][len(results)]))
        conversations.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": json.dumps(tool_call("calculate_math", arguments), ensure_ascii=False),
            }
        )
        conversations.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    if len(results) == 1:
        final_answer = f"计算结果是 {results[0]}。"
    else:
        final_answer = f"依次计算结果是 {'、'.join(results)}。"
    conversations.append({"role": "assistant", "content": final_answer})
    return {
        "id": f"{row['id']}_sft",
        "category": row["category"],
        "source": row["source"],
        "conversations": conversations,
    }


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_converted_items(
    sources: Sequence[Tuple[str, Iterable[Dict]]],
    max_source_rows: int = 0,
    prompt_mode: str = "original",
    include_single: bool = True,
    include_multi: bool = True,
    max_expressions: int = 3,
) -> Tuple[List[Dict], Dict[str, int]]:
    rows: List[Dict] = []
    stats = {
        "source_rows": 0,
        "converted_rows": 0,
        "empty_gt": 0,
        "non_empty_final": 0,
        "parse_or_verify_failed": 0,
        "duplicate_prompt": 0,
    }
    seen_prompts = set()
    for source, items in sources:
        for idx, row in enumerate(items, start=1):
            if max_source_rows and idx > max_source_rows:
                break
            stats["source_rows"] += 1
            conversations = row.get("conversations") or []
            if not row.get("gt"):
                stats["empty_gt"] += 1
                continue
            if not final_assistant_is_empty(conversations):
                stats["non_empty_final"] += 1
                continue
            converted_rows = convert_row_variants(
                row,
                source,
                idx,
                prompt_mode=prompt_mode,
                include_single=include_single,
                include_multi=include_multi,
                max_expressions=max_expressions,
            )
            if not converted_rows:
                stats["parse_or_verify_failed"] += 1
                continue
            for converted in converted_rows:
                if converted["prompt"] in seen_prompts:
                    stats["duplicate_prompt"] += 1
                    continue
                rows.append(converted)
                seen_prompts.add(converted["prompt"])
                stats["converted_rows"] += 1
    return rows, stats


def collect_converted_rows(
    paths: Sequence[Tuple[str, Path]],
    max_source_rows: int = 0,
    prompt_mode: str = "original",
    include_single: bool = True,
    include_multi: bool = True,
    max_expressions: int = 3,
) -> Tuple[List[Dict], Dict[str, int]]:
    sources = [(source, iter_jsonl(path)) for source, path in paths if path.exists()]
    return collect_converted_items(
        sources,
        max_source_rows=max_source_rows,
        prompt_mode=prompt_mode,
        include_single=include_single,
        include_multi=include_multi,
        max_expressions=max_expressions,
    )


def split_rows(rows: Sequence[Dict], eval_size: int, sft_seed_size: int, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    eval_rows = rows[: min(eval_size, len(rows))]
    sft_tasks = rows[len(eval_rows) : len(eval_rows) + min(sft_seed_size, max(0, len(rows) - len(eval_rows)))]
    rl_rows = rows[len(eval_rows) + len(sft_tasks) :]
    return eval_rows, [sft_sample_from_task(row) for row in sft_tasks], rl_rows


def summarize(rows: Sequence[Dict]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for row in rows:
        key = row.get("category", "unknown")
        summary[key] = summary.get(key, 0) + 1
    return summary


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original MiniMind Agent RL data into current tool-use RLVR data")
    parser.add_argument("--agent_rl_math_path", default="dataset/agent_rl_math.jsonl")
    parser.add_argument("--agent_rl_path", default="dataset/agent_rl.jsonl")
    parser.add_argument("--skip_agent_rl", action="store_true")
    parser.add_argument("--eval_size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--sft_seed_size", type=int, default=DEFAULT_SFT_SEED_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_source_rows", type=int, default=0)
    parser.add_argument("--prompt_mode", choices=PROMPT_MODES, default="original")
    parser.add_argument("--include_single", action="store_true")
    parser.add_argument("--include_multi", action="store_true")
    parser.add_argument("--max_expressions", type=int, default=3)
    parser.add_argument("--eval_path", default="evals/tool_eval_minimind_math.jsonl")
    parser.add_argument("--sft_seed_path", default="dataset/tool_sft_minimind_math_seed.jsonl")
    parser.add_argument("--rl_train_path", default="dataset/agent_rl_tooluse_minimind_math_train.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    include_single = args.include_single or not args.include_multi
    include_multi = args.include_multi or not args.include_single

    sources = [("agent_rl_math", resolve_path(args.agent_rl_math_path))]
    if not args.skip_agent_rl:
        sources.append(("agent_rl", resolve_path(args.agent_rl_path)))

    rows, stats = collect_converted_rows(
        sources,
        max_source_rows=args.max_source_rows,
        prompt_mode=args.prompt_mode,
        include_single=include_single,
        include_multi=include_multi,
        max_expressions=args.max_expressions,
    )
    eval_rows, sft_rows, rl_rows = split_rows(rows, args.eval_size, args.sft_seed_size, args.seed)

    print("conversion stats:", json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"verified rows: {len(rows)} {summarize(rows)}")
    print(f"eval rows: {len(eval_rows)} {summarize(eval_rows)}")
    print(f"sft seed rows: {len(sft_rows)}")
    print(f"rl train rows: {len(rl_rows)} {summarize(rl_rows)}")
    print("prompt overlap: none")

    if args.dry_run:
        print("dry run: no files written")
        return

    outputs = [
        (resolve_path(args.eval_path), eval_rows),
        (resolve_path(args.sft_seed_path), sft_rows),
        (resolve_path(args.rl_train_path), rl_rows),
    ]
    for path, output_rows in outputs:
        write_jsonl(path, output_rows)
        print(f"wrote {len(output_rows)} rows -> {path}")


if __name__ == "__main__":
    main()
