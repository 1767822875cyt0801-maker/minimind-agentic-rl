#!/bin/bash
set -euo pipefail

# Stage-4 RL wrapper.
# Continue the comparable scale-up series from the clean v13 SFT checkpoint.

export STAGE_LABEL=${STAGE_LABEL:-stage4}
export EXP_NAME=${EXP_NAME:-v13_rl_stage4_$(date +%Y%m%d_%H%M%S)}
export FROM_WEIGHT=${FROM_WEIGHT:-tool_sft_v13_stop_repair_repeated_math}
export SAVE_WEIGHT=${SAVE_WEIGHT:-agent_tool_v13_rl_norm_stage4}
export RL_ROWS=${RL_ROWS:-2048}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export MAX_TOOL_TURNS=${MAX_TOOL_TURNS:-4}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export RL_LEARNING_RATE=${RL_LEARNING_RATE:-5e-8}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_v13_rl_stage1_autodl.sh"
