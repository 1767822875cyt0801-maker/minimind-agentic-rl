#!/bin/bash
set -e

EXP_NAME=full_sft_$(date +%Y%m%d_%H%M%S)

CODE_ROOT=/root/minimind
FS_ROOT=/root/autodl-fs/minimind
TMP_ROOT=/root/autodl-tmp/minimind

SFT_DATA_SRC=$FS_ROOT/datasets/sft_t2t_mini.jsonl
PRETRAIN_SRC=$FS_ROOT/checkpoints/pretrain_768_backup/pretrain_768.pth

SFT_DATA_DST=$TMP_ROOT/datasets/sft_t2t_mini.jsonl
PRETRAIN_DST=$TMP_ROOT/checkpoints/pretrain_768.pth

RUN_DIR=$TMP_ROOT/runs/$EXP_NAME
OUT_DIR=$RUN_DIR/out
LOG_DIR=$RUN_DIR/logs

mkdir -p $TMP_ROOT/datasets
mkdir -p $TMP_ROOT/checkpoints
mkdir -p $TMP_ROOT/runs
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

mkdir -p $FS_ROOT/checkpoints
mkdir -p $FS_ROOT/logs
mkdir -p $FS_ROOT/runs

echo "==== [1] copy dataset and pretrain weight to local SSD ===="
cp $SFT_DATA_SRC $SFT_DATA_DST
cp $PRETRAIN_SRC $PRETRAIN_DST

echo "==== [2] start full sft ===="
cd $CODE_ROOT/trainer

python train_full_sft.py \
  --save_dir $OUT_DIR \
  --save_weight full_sft \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_workers 8 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 1000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --use_moe 0 \
  --data_path $SFT_DATA_DST \
  --from_weight $PRETRAIN_DST \
  --from_resume 0 \
  --use_compile 0 \
  2>&1 | tee $LOG_DIR/train.log

echo "==== [3] backup outputs ===="
cp -r $OUT_DIR $FS_ROOT/runs/$EXP_NAME
cp $LOG_DIR/train.log $FS_ROOT/logs/${EXP_NAME}.log

echo "==== [4] backup checkpoints saved by trainer ===="
find /root/minimind/checkpoints -maxdepth 1 -type f \( -name "full_sft*.pth" -o -name "*resume*.pth" \) -exec cp {} $FS_ROOT/checkpoints/ \;

echo "==== done ===="
echo "run dir: $RUN_DIR"
echo "backup runs: $FS_ROOT/runs/$EXP_NAME"
echo "backup checkpoints: $FS_ROOT/checkpoints"
