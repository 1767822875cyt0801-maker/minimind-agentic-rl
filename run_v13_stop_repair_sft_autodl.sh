#!/bin/bash
set -euo pipefail

# v13: targeted stop-after-observation repair.
# Use after v12 improves hard but still leaves basic/normalized tool loops.

EXP_NAME=${EXP_NAME:-v13_stop_repair_$(date +%Y%m%d_%H%M%S)}

CODE_ROOT=${CODE_ROOT:-/root/minimind}
FS_ROOT=${FS_ROOT:-/root/autodl-fs/minimind}
TMP_ROOT=${TMP_ROOT:-/root/autodl-tmp/minimind}

DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-8}
USE_MOE=${USE_MOE:-0}

FROM_WEIGHT=${FROM_WEIGHT:-tool_sft_v12_oldhard_repair_repeated_math}
FALLBACK_FROM_WEIGHT=${FALLBACK_FROM_WEIGHT:-tool_sft_v11_balanced_repeated_math}
V13_WEIGHT=${V13_WEIGHT:-tool_sft_v13_stop_repair_repeated_math}
V13_DATASET=${V13_DATASET:-dataset/tool_sft_v13_stop_repair_repeated_math_mixed.jsonl}
V13_RAW_DATASET=${V13_RAW_DATASET:-dataset/tool_sft_v13_stop_repair_repeated_math_mixed_raw.jsonl}
NORMALIZED_EVAL_PATH=${NORMALIZED_EVAL_PATH:-evals/tool_eval_minimind_math_normalized.jsonl}

PATCH_REPEATS=${PATCH_REPEATS:-10}
HARD_REPEATS=${HARD_REPEATS:-10}
TIME_REPEATS=${TIME_REPEATS:-4}
DYNAMIC_REPEATS=${DYNAMIC_REPEATS:-1}
STOP_REPAIR_REPEATS=${STOP_REPAIR_REPEATS:-2}
STOP_REPAIR_ROWS=${STOP_REPAIR_ROWS:-1200}
NORMALIZED_SEED_SIZE=${NORMALIZED_SEED_SIZE:-500}
NORMALIZED_SEED_TAG=${NORMALIZED_SEED_TAG:-$NORMALIZED_SEED_SIZE}
LEARNING_RATE=${LEARNING_RATE:-7e-8}

RUN_DIR=$TMP_ROOT/runs/$EXP_NAME
LOG_DIR=$RUN_DIR/logs
REPORT_DIR=$CODE_ROOT/evals/reports/$V13_WEIGHT
GENERATED_EVAL_DIR=$CODE_ROOT/evals/generated/v13_prep

RUN_RL_SMOKE=${RUN_RL_SMOKE:-0}
RL_SMOKE_ROWS=${RL_SMOKE_ROWS:-32}
RL_SMOKE_WEIGHT=${RL_SMOKE_WEIGHT:-agent_tool_v13_smoke}
MAX_TOOL_TURNS=${MAX_TOOL_TURNS:-4}

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$GENERATED_EVAL_DIR"
mkdir -p "$FS_ROOT/checkpoints" "$FS_ROOT/datasets" "$FS_ROOT/logs" "$FS_ROOT/reports" "$FS_ROOT/runs"
mkdir -p "$CODE_ROOT/out" "$CODE_ROOT/dataset" "$CODE_ROOT/evals"

cd "$CODE_ROOT"

echo "==== [1] preflight: code, source weight, required data ===="
test -f trainer/train_full_sft.py
test -f scripts/convert_minimind_agent_rl_data.py
test -f scripts/make_tool_sft_stop_after_observation_repair.py

ensure_weight_in_out() {
  local weight_name="$1"
  local out_path="out/${weight_name}_${HIDDEN_SIZE}.pth"
  if [ ! -f "$out_path" ]; then
    for candidate in \
      "checkpoints/${weight_name}_${HIDDEN_SIZE}.pth" \
      "$FS_ROOT/checkpoints/${weight_name}_${HIDDEN_SIZE}.pth" \
      "$TMP_ROOT/checkpoints/${weight_name}_${HIDDEN_SIZE}.pth"
    do
      if [ -f "$candidate" ]; then
        echo "Found $weight_name checkpoint at $candidate; copying to $out_path"
        cp "$candidate" "$out_path"
        break
      fi
    done
  fi
  test -f "$out_path"
}

if ! ensure_weight_in_out "$FROM_WEIGHT"; then
  echo "Source weight $FROM_WEIGHT is missing; trying fallback $FALLBACK_FROM_WEIGHT."
  FROM_WEIGHT="$FALLBACK_FROM_WEIGHT"
fi

if ! ensure_weight_in_out "$FROM_WEIGHT"; then
  echo "MISSING: $CODE_ROOT/out/${FROM_WEIGHT}_${HIDDEN_SIZE}.pth"
  echo "Place the source checkpoint under $CODE_ROOT/out before v13 SFT."
  exit 2
fi

if [ ! -f dataset/tool_sft_hard_train.jsonl ]; then
  echo "dataset/tool_sft_hard_train.jsonl is missing; regenerating hard SFT train data."
  python scripts/make_agentic_tooluse_hard_data.py \
    --preset full \
    --eval_path "$GENERATED_EVAL_DIR/tool_eval_hard_from_v13_prep.jsonl" \
    --sft_path dataset/tool_sft_hard_train.jsonl \
    --rl_path dataset/agent_rl_tooluse_hard_train.jsonl
fi

if [ ! -f dataset/tool_sft_dynamic_zh_seed.jsonl ]; then
  echo "dataset/tool_sft_dynamic_zh_seed.jsonl is missing; regenerating dynamic zh SFT seed."
  python scripts/make_agentic_tooluse_dynamic_data.py \
    --preset full \
    --language zh \
    --l1_path "$GENERATED_EVAL_DIR/tool_eval_dynamic_zh_l1_from_v13_prep.jsonl" \
    --l2_path "$GENERATED_EVAL_DIR/tool_eval_dynamic_zh_l2_from_v13_prep.jsonl" \
    --l3_path "$GENERATED_EVAL_DIR/tool_eval_dynamic_zh_l3_from_v13_prep.jsonl" \
    --l4_path "$GENERATED_EVAL_DIR/tool_eval_dynamic_zh_l4_from_v13_prep.jsonl" \
    --train_path dataset/agent_rl_tooluse_dynamic_zh_train.jsonl \
    --sft_seed_path dataset/tool_sft_dynamic_zh_seed.jsonl
fi

if [ ! -f dataset/tool_sft_small_v4_patch.jsonl ] && [ "${ALLOW_PATCH_V4_FALLBACK:-0}" = "1" ]; then
  echo "ALLOW_PATCH_V4_FALLBACK=1: generating tool_sft_patch_v4.jsonl and using it as tool_sft_small_v4_patch.jsonl."
  python scripts/make_tool_sft_patch_v4.py
  cp dataset/tool_sft_patch_v4.jsonl dataset/tool_sft_small_v4_patch.jsonl
fi

missing_required=0
for f in \
  dataset/tool_sft_small_v4_patch.jsonl \
  dataset/tool_sft_hard_train.jsonl \
  dataset/tool_sft_v6_time_obs_repair.jsonl \
  dataset/tool_sft_dynamic_zh_seed.jsonl
do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    missing_required=1
  fi
done

if [ "$missing_required" -ne 0 ]; then
  echo "Restore the missing historical SFT files before training."
  echo "Do not substitute evals/tool_eval*.jsonl into the SFT mixture."
  exit 2
fi

echo "==== [2] generate ${NORMALIZED_SEED_SIZE} normalized multi-call seed ===="
python scripts/convert_minimind_agent_rl_data.py \
  --skip_agent_rl \
  --prompt_mode normalized \
  --include_multi \
  --sft_seed_size "$NORMALIZED_SEED_SIZE" \
  --sft_seed_path "dataset/tool_sft_minimind_math_normalized_seed_${NORMALIZED_SEED_TAG}.jsonl" \
  --eval_path "evals/tool_eval_minimind_math_normalized_${NORMALIZED_SEED_TAG}_split.jsonl" \
  --rl_train_path "dataset/agent_rl_tooluse_minimind_math_normalized_${NORMALIZED_SEED_TAG}_train.jsonl" \
  2>&1 | tee "$LOG_DIR/convert_normalized_${NORMALIZED_SEED_TAG}.log"

if [ ! -f "$NORMALIZED_EVAL_PATH" ]; then
  echo "$NORMALIZED_EVAL_PATH is missing; using the generated normalized eval split for v13 evaluation."
  cp "evals/tool_eval_minimind_math_normalized_${NORMALIZED_SEED_TAG}_split.jsonl" "$NORMALIZED_EVAL_PATH"
fi

echo "==== [3] generate stop-after-observation repair seed ===="
python scripts/make_tool_sft_stop_after_observation_repair.py \
  --rows "$STOP_REPAIR_ROWS" \
  --output dataset/tool_sft_stop_after_observation_repair.jsonl \
  2>&1 | tee "$LOG_DIR/make_stop_repair.log"

echo "==== [4] build v13 targeted repair SFT mixture ===="
: > "$V13_RAW_DATASET"
for _ in $(seq 1 "$PATCH_REPEATS"); do
  cat dataset/tool_sft_small_v4_patch.jsonl >> "$V13_RAW_DATASET"
done
for _ in $(seq 1 "$HARD_REPEATS"); do
  cat dataset/tool_sft_hard_train.jsonl >> "$V13_RAW_DATASET"
done
for _ in $(seq 1 "$TIME_REPEATS"); do
  cat dataset/tool_sft_v6_time_obs_repair.jsonl >> "$V13_RAW_DATASET"
done
for _ in $(seq 1 "$DYNAMIC_REPEATS"); do
  cat dataset/tool_sft_dynamic_zh_seed.jsonl >> "$V13_RAW_DATASET"
done
for _ in $(seq 1 "$STOP_REPAIR_REPEATS"); do
  cat dataset/tool_sft_stop_after_observation_repair.jsonl >> "$V13_RAW_DATASET"
done
cat "dataset/tool_sft_minimind_math_normalized_seed_${NORMALIZED_SEED_TAG}.jsonl" >> "$V13_RAW_DATASET"

python - "$V13_RAW_DATASET" "$V13_DATASET" "$PATCH_REPEATS" "$HARD_REPEATS" "$TIME_REPEATS" "$DYNAMIC_REPEATS" "$STOP_REPAIR_REPEATS" "$STOP_REPAIR_ROWS" "$NORMALIZED_SEED_SIZE" <<'PY' | tee "$LOG_DIR/v13_dataset_summary.log"
import json
import sys
from collections import Counter
from pathlib import Path

raw_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
print("mixture_repeats:")
print(f"  patch: {sys.argv[3]}")
print(f"  hard: {sys.argv[4]}")
print(f"  time: {sys.argv[5]}")
print(f"  dynamic: {sys.argv[6]}")
print(f"  stop_repair: {sys.argv[7]}")
print(f"  stop_repair_rows: {sys.argv[8]}")
print(f"  normalized_seed_size: {sys.argv[9]}")

counts = Counter()
top_level_keys = Counter()
rows = 0
with raw_path.open("r", encoding="utf-8") as f_in, train_path.open("w", encoding="utf-8", newline="\n") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        if "conversations" not in row:
            raise ValueError(f"row {rows} is missing conversations")
        counts[row.get("category", "unknown")] += 1
        top_level_keys[tuple(sorted(row.keys()))] += 1
        f_out.write(json.dumps({"conversations": row["conversations"]}, ensure_ascii=False) + "\n")

print(f"raw_dataset: {raw_path}")
print(f"train_dataset: {train_path}")
print(f"rows: {rows}")
print("categories:")
for key, value in sorted(counts.items()):
    print(f"  {key}: {value}")
print("top_level_schema_variants_before_normalization:")
for keys, value in top_level_keys.most_common():
    print(f"  {value}: {list(keys)}")
print("normalized train rows contain only the conversations column")
PY

echo "==== [5] train $V13_WEIGHT from $FROM_WEIGHT ===="
cd "$CODE_ROOT/trainer"

OMP_NUM_THREADS=1 python train_full_sft.py \
  --save_dir ../out \
  --from_weight "$FROM_WEIGHT" \
  --save_weight "$V13_WEIGHT" \
  --data_path "../$V13_DATASET" \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate "$LEARNING_RATE" \
  --max_seq_len 1536 \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num_workers 0 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 300 \
  --hidden_size "$HIDDEN_SIZE" \
  --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
  --use_moe "$USE_MOE" \
  --from_resume 0 \
  --use_compile 0 \
  2>&1 | tee "$LOG_DIR/train_v13.log"

cd "$CODE_ROOT"

echo "==== [6] evaluate basic / hard / normalized multi ===="
python evals/run_tool_eval.py \
  --weight "$V13_WEIGHT" \
  --mode guarded \
  --eval_path evals/tool_eval.jsonl \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/basic_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_basic.log"

python evals/run_tool_eval.py \
  --weight "$V13_WEIGHT" \
  --mode guarded \
  --eval_path evals/tool_eval_hard.jsonl \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/hard_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_hard.log"

python evals/run_tool_eval.py \
  --weight "$V13_WEIGHT" \
  --mode guarded \
  --eval_path "$NORMALIZED_EVAL_PATH" \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/minimind_math_normalized_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_minimind_math_normalized.log"

echo "==== [7] summarize and gate RL smoke ===="
python - "$REPORT_DIR" <<'PY' | tee "$LOG_DIR/v13_eval_summary.log"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
names = ["basic", "hard", "minimind_math_normalized"]
reports = {}
for name in names:
    path = base / f"{name}_guarded.json"
    reports[name] = json.loads(path.read_text(encoding="utf-8"))

for name in names:
    r = reports[name]
    print("=" * 80)
    print(name)
    print("valid:", round(r.get("valid_rate", 0), 4))
    print("tool :", round(r.get("tool_acc", 0), 4))
    print("args :", round(r.get("args_acc", 0), 4))
    print("obs  :", round(r.get("obs_use_acc", 0), 4))
    print("ans  :", round(r.get("answer_acc", 0), 4))
    print("loop :", round(r.get("loop_rate", 0), 4))
    print("mal  :", round(r.get("malformed_rate", 0), 4))
    print("unfin:", round(r.get("unfinished_rate", 0), 4))
    print("reward:", round(r.get("avg_reward", 0), 4))

checks = {
    "basic_avg_reward": reports["basic"].get("avg_reward", 0) >= 2.95,
    "hard_avg_reward": reports["hard"].get("avg_reward", 0) >= 2.94,
    "hard_answer_acc": reports["hard"].get("answer_acc", 0) >= 0.97,
    "normalized_tool_acc": reports["minimind_math_normalized"].get("tool_acc", 0) >= 0.60,
    "normalized_answer_acc": reports["minimind_math_normalized"].get("answer_acc", 0) >= 0.90,
}
for name in names:
    checks[f"{name}_malformed_zero"] = reports[name].get("malformed_rate", 1) == 0
    checks[f"{name}_loop_zero"] = reports[name].get("loop_rate", 1) == 0
    checks[f"{name}_unfinished_zero"] = reports[name].get("unfinished_rate", 1) == 0

passed = all(checks.values())
if passed:
    next_step = "Run RL smoke only; do not start full RL yet."
elif reports["hard"].get("avg_reward", 0) < 2.94:
    next_step = "Do not run RL. Inspect v13 failures; consider another small stop-repair pass or lower normalized seed."
elif reports["minimind_math_normalized"].get("tool_acc", 0) < 0.60:
    next_step = "Do not run RL. Re-add normalized seed to 1000-1500 while keeping stop-repair rows."
else:
    next_step = "Do not run RL. Inspect residual loop failures before another SFT repair."

verdict = {"passed": passed, "checks": checks, "next_step": next_step}
(base / "v13_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
(base / "v13_verdict.env").write_text(f"V13_PASS={1 if passed else 0}\n", encoding="utf-8")

print("=" * 80)
print("V13_PASS:", int(passed))
for key, ok in checks.items():
    print(f"{key}: {'PASS' if ok else 'FAIL'}")
print("next_step:", next_step)
PY

# shellcheck disable=SC1091
source "$REPORT_DIR/v13_verdict.env"

if [ "$V13_PASS" = "1" ] && [ "$RUN_RL_SMOKE" = "1" ]; then
  echo "==== [8] optional RL smoke ===="
  RL_SMOKE_DATA="dataset/agent_rl_tooluse_minimind_math_normalized_${NORMALIZED_SEED_TAG}_smoke.jsonl"
  python - "$RL_SMOKE_ROWS" "dataset/agent_rl_tooluse_minimind_math_normalized_${NORMALIZED_SEED_TAG}_train.jsonl" "$RL_SMOKE_DATA" <<'PY'
import sys
from pathlib import Path

limit = int(sys.argv[1])
src = Path(sys.argv[2])
dst = Path(sys.argv[3])
dst.parent.mkdir(parents=True, exist_ok=True)
rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(line)
        if len(rows) >= limit:
            break
dst.write_text("".join(rows), encoding="utf-8")
print(f"wrote {len(rows)} rows -> {dst}")
PY

  cd "$CODE_ROOT/trainer"
  OMP_NUM_THREADS=1 python train_agent.py \
    --rollout_engine torch \
    --from_weight "$V13_WEIGHT" \
    --save_weight "$RL_SMOKE_WEIGHT" \
    --data_path "../$RL_SMOKE_DATA" \
    --batch_size 1 \
    --num_generations 4 \
    --tool_env_mode guarded \
    --rollout_temperature 0.8 \
    --max_tool_turns "$MAX_TOOL_TURNS" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --epochs 1 \
    --learning_rate 1e-7 \
    --max_seq_len 1536 \
    --max_gen_len 384 \
    --max_total_len 2500 \
    --thinking_ratio 0.0 \
    --save_rollout_trace_path "$REPORT_DIR/rl_smoke_rollout_trace.jsonl" \
    --log_interval 1 \
    --save_interval 20 \
    --num_workers 0 \
    --from_resume 0 \
    --use_compile 0 \
    2>&1 | tee "$LOG_DIR/rl_smoke.log"
  cd "$CODE_ROOT"
else
  echo "RL smoke skipped. Set RUN_RL_SMOKE=1 and pass v13 thresholds to run it."
fi

echo "==== [9] backup artifacts ===="
cp "$CODE_ROOT/out/${V13_WEIGHT}_${HIDDEN_SIZE}.pth" "$FS_ROOT/checkpoints/" || true
cp "$CODE_ROOT/checkpoints/${V13_WEIGHT}_${HIDDEN_SIZE}.pth" "$FS_ROOT/checkpoints/" 2>/dev/null || true
cp "$CODE_ROOT/checkpoints/${V13_WEIGHT}_${HIDDEN_SIZE}_resume.pth" "$FS_ROOT/checkpoints/" 2>/dev/null || true
cp "$CODE_ROOT/$V13_DATASET" "$FS_ROOT/datasets/" || true
cp "$CODE_ROOT/$V13_RAW_DATASET" "$FS_ROOT/datasets/" || true
cp "$LOG_DIR"/*.log "$FS_ROOT/logs/" || true
mkdir -p "$FS_ROOT/reports/$V13_WEIGHT"
cp "$REPORT_DIR"/* "$FS_ROOT/reports/$V13_WEIGHT/" || true

echo "==== done ===="
echo "run dir: $RUN_DIR"
echo "logs: $LOG_DIR"
echo "reports: $REPORT_DIR"
echo "source weight: $FROM_WEIGHT"
echo "v13 checkpoint: $CODE_ROOT/out/${V13_WEIGHT}_${HIDDEN_SIZE}.pth"
echo "backup checkpoints: $FS_ROOT/checkpoints"
echo "backup reports: $FS_ROOT/reports/$V13_WEIGHT"
