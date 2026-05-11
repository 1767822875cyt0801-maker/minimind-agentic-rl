#!/bin/bash
set -euo pipefail

# Stage-1 RL after v13 variance probe.
# Start from the clean v13 SFT checkpoint, not from a probe/smoke checkpoint.

EXP_NAME=${EXP_NAME:-v13_rl_stage1_$(date +%Y%m%d_%H%M%S)}
STAGE_LABEL=${STAGE_LABEL:-stage1}

CODE_ROOT=${CODE_ROOT:-/root/minimind}
FS_ROOT=${FS_ROOT:-/root/autodl-fs/minimind}
TMP_ROOT=${TMP_ROOT:-/root/autodl-tmp/minimind}

DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-8}
USE_MOE=${USE_MOE:-0}

FROM_WEIGHT=${FROM_WEIGHT:-tool_sft_v13_stop_repair_repeated_math}
SAVE_WEIGHT=${SAVE_WEIGHT:-agent_tool_v13_rl_norm_stage1}
BASELINE_REPORT_WEIGHT=${BASELINE_REPORT_WEIGHT:-tool_sft_v13_stop_repair_repeated_math}
NORMALIZED_SEED_TAG=${NORMALIZED_SEED_TAG:-500}
NORMALIZED_RL_TRAIN=${NORMALIZED_RL_TRAIN:-dataset/agent_rl_tooluse_minimind_math_normalized_${NORMALIZED_SEED_TAG}_train.jsonl}
NORMALIZED_EVAL_PATH=${NORMALIZED_EVAL_PATH:-evals/tool_eval_minimind_math_normalized.jsonl}

RL_ROWS=${RL_ROWS:-128}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_TOOL_TURNS=${MAX_TOOL_TURNS:-4}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
RL_LEARNING_RATE=${RL_LEARNING_RATE:-5e-8}
LOSS_TYPE=${LOSS_TYPE:-cispo}
BETA=${BETA:-0.1}

RUN_DIR=$TMP_ROOT/runs/$EXP_NAME
LOG_DIR=$RUN_DIR/logs
REPORT_DIR=$CODE_ROOT/evals/reports/$SAVE_WEIGHT
BASELINE_REPORT_DIR=$CODE_ROOT/evals/reports/$BASELINE_REPORT_WEIGHT
TRACE_PATH=$REPORT_DIR/rl_${STAGE_LABEL}_rollout_trace.jsonl
RL_DATASET=dataset/agent_rl_tooluse_minimind_math_normalized_${NORMALIZED_SEED_TAG}_${STAGE_LABEL}_${RL_ROWS}.jsonl

mkdir -p "$LOG_DIR" "$REPORT_DIR"
mkdir -p "$FS_ROOT/checkpoints" "$FS_ROOT/datasets" "$FS_ROOT/logs" "$FS_ROOT/reports" "$FS_ROOT/runs"
mkdir -p "$CODE_ROOT/out" "$CODE_ROOT/dataset" "$CODE_ROOT/evals"

cd "$CODE_ROOT"

echo "==== [1] preflight: v13 source weight and baseline reports ===="
test -f trainer/train_agent.py
test -f evals/run_tool_eval.py
test -f scripts/analyze_rollout_trace.py

python - <<'PY'
import sys

try:
    import numpy
    import scipy
    import sklearn
    from transformers import AutoTokenizer  # noqa: F401
except Exception as exc:
    print("PYTHON_DEP_IMPORT_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    print()
    print("This is an AutoDL Python environment issue, usually a numpy/scipy/sklearn")
    print("binary compatibility mismatch triggered while transformers imports sklearn.")
    print("Fix it once in the container, then rerun this script:")
    print()
    print('  python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "scipy==1.12.0" "scikit-learn==1.5.1"')
    print("  python - <<'PYCHK'")
    print("  import numpy, scipy, sklearn")
    print("  from transformers import AutoTokenizer")
    print("  print('deps ok', numpy.__version__, scipy.__version__, sklearn.__version__)")
    print("  PYCHK")
    sys.exit(86)

print(
    "python deps OK:",
    f"numpy={numpy.__version__}",
    f"scipy={scipy.__version__}",
    f"sklearn={sklearn.__version__}",
)
PY

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
  echo "MISSING: $CODE_ROOT/out/${FROM_WEIGHT}_${HIDDEN_SIZE}.pth"
  exit 2
fi

for f in \
  "$BASELINE_REPORT_DIR/basic_guarded.json" \
  "$BASELINE_REPORT_DIR/hard_guarded.json" \
  "$BASELINE_REPORT_DIR/minimind_math_normalized_guarded.json"
do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    echo "Run v13 SFT eval first so stage1 can compare against the baseline."
    exit 2
  fi
done

if [ ! -f "$NORMALIZED_RL_TRAIN" ]; then
  echo "MISSING: $NORMALIZED_RL_TRAIN"
  echo "Run v13 SFT script first, or regenerate normalized RL train with convert_minimind_agent_rl_data.py."
  exit 2
fi

echo "==== [2] prepare $STAGE_LABEL RL dataset ===="
python - "$RL_ROWS" "$NORMALIZED_RL_TRAIN" "$RL_DATASET" <<'PY'
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

rm -f "$TRACE_PATH"

echo "==== [3] run $STAGE_LABEL RL from $FROM_WEIGHT ===="
cd "$CODE_ROOT/trainer"
OMP_NUM_THREADS=1 python train_agent.py \
  --rollout_engine torch \
  --from_weight "$FROM_WEIGHT" \
  --save_weight "$SAVE_WEIGHT" \
  --data_path "../$RL_DATASET" \
  --batch_size 1 \
  --num_generations "$NUM_GENERATIONS" \
  --loss_type "$LOSS_TYPE" \
  --beta "$BETA" \
  --tool_env_mode guarded \
  --rollout_temperature "$ROLLOUT_TEMPERATURE" \
  --rollout_top_p "$ROLLOUT_TOP_P" \
  --max_tool_turns "$MAX_TOOL_TURNS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --epochs 1 \
  --learning_rate "$RL_LEARNING_RATE" \
  --max_seq_len 1536 \
  --max_gen_len 384 \
  --max_total_len 2500 \
  --thinking_ratio 0.0 \
  --save_rollout_trace_path "$TRACE_PATH" \
  --log_interval 1 \
  --save_interval 20 \
  --hidden_size "$HIDDEN_SIZE" \
  --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
  --use_moe "$USE_MOE" \
  --num_workers 0 \
  --from_resume 0 \
  --use_compile 0 \
  2>&1 | tee "$LOG_DIR/rl_${STAGE_LABEL}.log"
cd "$CODE_ROOT"

echo "==== [4] summarize rollout trace ===="
python scripts/analyze_rollout_trace.py "$TRACE_PATH" \
  --output "$REPORT_DIR/rl_${STAGE_LABEL}_trace_summary.json" \
  2>&1 | tee "$LOG_DIR/rl_${STAGE_LABEL}_trace_summary.log"

echo "==== [5] evaluate $STAGE_LABEL checkpoint ===="
python evals/run_tool_eval.py \
  --weight "$SAVE_WEIGHT" \
  --mode guarded \
  --eval_path evals/tool_eval.jsonl \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/basic_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_basic.log"

python evals/run_tool_eval.py \
  --weight "$SAVE_WEIGHT" \
  --mode guarded \
  --eval_path evals/tool_eval_hard.jsonl \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/hard_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_hard.log"

python evals/run_tool_eval.py \
  --weight "$SAVE_WEIGHT" \
  --mode guarded \
  --eval_path "$NORMALIZED_EVAL_PATH" \
  --device "$DEVICE" \
  --continue_on_error \
  --output "$REPORT_DIR/minimind_math_normalized_guarded.json" \
  2>&1 | tee "$LOG_DIR/eval_minimind_math_normalized.log"

echo "==== [6] compare $STAGE_LABEL against v13 baseline ===="
python - "$BASELINE_REPORT_DIR" "$REPORT_DIR" "$STAGE_LABEL" <<'PY' | tee "$LOG_DIR/${STAGE_LABEL}_compare.log"
import json
import sys
from pathlib import Path

baseline_dir = Path(sys.argv[1])
stage_dir = Path(sys.argv[2])
stage_label = sys.argv[3]
names = ["basic", "hard", "minimind_math_normalized"]
metrics = ["valid_rate", "tool_acc", "args_acc", "answer_acc", "loop_rate", "malformed_rate", "unfinished_rate", "avg_reward"]

baseline = {name: json.loads((baseline_dir / f"{name}_guarded.json").read_text(encoding="utf-8")) for name in names}
stage = {name: json.loads((stage_dir / f"{name}_guarded.json").read_text(encoding="utf-8")) for name in names}

for name in names:
    print("=" * 80)
    print(name)
    for metric in metrics:
        b = float(baseline[name].get(metric, 0))
        s = float(stage[name].get(metric, 0))
        print(f"{metric}: baseline={b:.4f} {stage_label}={s:.4f} delta={s-b:+.4f}")

checks = {
    "basic_reward": stage["basic"].get("avg_reward", 0) >= 2.95,
    "basic_loop_zero": stage["basic"].get("loop_rate", 1) == 0,
    "basic_malformed_zero": stage["basic"].get("malformed_rate", 1) == 0,
    "basic_unfinished_zero": stage["basic"].get("unfinished_rate", 1) == 0,
    "hard_reward": stage["hard"].get("avg_reward", 0) >= 2.94,
    "hard_answer": stage["hard"].get("answer_acc", 0) >= 0.97,
    "hard_loop_zero": stage["hard"].get("loop_rate", 1) == 0,
    "hard_malformed_zero": stage["hard"].get("malformed_rate", 1) == 0,
    "hard_unfinished_zero": stage["hard"].get("unfinished_rate", 1) == 0,
    "normalized_tool": stage["minimind_math_normalized"].get("tool_acc", 0) >= 0.60,
    "normalized_answer": stage["minimind_math_normalized"].get("answer_acc", 0) >= 0.90,
    "normalized_loop_not_worse": stage["minimind_math_normalized"].get("loop_rate", 1)
    <= baseline["minimind_math_normalized"].get("loop_rate", 1) + 0.005,
    "normalized_reward_not_bad": stage["minimind_math_normalized"].get("avg_reward", 0)
    >= baseline["minimind_math_normalized"].get("avg_reward", 0) - 0.05,
}
passed = all(checks.values())
print("=" * 80)
print(f"{stage_label.upper()}_PASS:", int(passed))
for key, ok in checks.items():
    print(f"{key}: {'PASS' if ok else 'FAIL'}")

verdict = {
    "passed": passed,
    "checks": checks,
    "next_step": (
        "Consider the next wider RL stage from the original v13 SFT checkpoint."
        if passed
        else f"Do not continue RL. Inspect {stage_label} regressions and consider a smaller LR or harder filtered RL data."
    ),
}
(stage_dir / f"{stage_label}_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "==== [7] backup artifacts ===="
cp "$CODE_ROOT/out/${SAVE_WEIGHT}_${HIDDEN_SIZE}.pth" "$FS_ROOT/checkpoints/" || true
cp "$CODE_ROOT/checkpoints/${SAVE_WEIGHT}_${HIDDEN_SIZE}.pth" "$FS_ROOT/checkpoints/" 2>/dev/null || true
cp "$CODE_ROOT/checkpoints/${SAVE_WEIGHT}_${HIDDEN_SIZE}_resume.pth" "$FS_ROOT/checkpoints/" 2>/dev/null || true
cp "$CODE_ROOT/$RL_DATASET" "$FS_ROOT/datasets/" || true
cp "$LOG_DIR"/*.log "$FS_ROOT/logs/" || true
mkdir -p "$FS_ROOT/reports/$SAVE_WEIGHT"
cp "$REPORT_DIR"/* "$FS_ROOT/reports/$SAVE_WEIGHT/" || true

echo "==== done ===="
echo "run dir: $RUN_DIR"
echo "logs: $LOG_DIR"
echo "reports: $REPORT_DIR"
echo "trace: $TRACE_PATH"
echo "$STAGE_LABEL checkpoint: $CODE_ROOT/out/${SAVE_WEIGHT}_${HIDDEN_SIZE}.pth"
