#!/usr/bin/env bash
set -euo pipefail

# Reproduce Cloud-Stereo (BMVC 2025) style protocol in UniMatch:
#   1) Fine-tune on synthetic split
#   2) Evaluate on real-world split with sparse LiDAR disparity

NUM_GPUS=${NUM_GPUS:-8}
DATA_ROOT=${DATA_ROOT:-datasets/cloud-stereo}
TRAIN_JSON=${TRAIN_JSON:-train.json}
VAL_JSON=${VAL_JSON:-val.json}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints_stereo/cloudstereo-bmvc2025-gmstereo}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-pretrained/gmflow-scale2-things-36579974.pth}
MASTER_PORT=${MASTER_PORT:-9989}

if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "Missing pretrained checkpoint: ${PRETRAINED_CKPT}"
  echo "Set PRETRAINED_CKPT=/path/to/ckpt.pth and re-run."
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

python -m torch.distributed.launch --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" main_stereo.py \
  --launcher pytorch \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --resume "${PRETRAINED_CKPT}" \
  --no_resume_optimizer \
  --stage cloud_stereo \
  --cloudstereo_root "${DATA_ROOT}" \
  --cloudstereo_train_json "${TRAIN_JSON}" \
  --cloudstereo_val_json "${VAL_JSON}" \
  --val_dataset cloudstereo \
  --img_height 384 \
  --img_width 768 \
  --padding_factor 32 \
  --upsample_factor 4 \
  --num_scales 2 \
  --attn_type self_swin2d_cross_swin1d \
  --attn_splits_list 2 8 \
  --corr_radius_list -1 4 \
  --prop_radius_list -1 1 \
  --batch_size 32 \
  --summary_freq 100 \
  --val_freq 5000 \
  --save_ckpt_freq 1000 \
  --save_latest_ckpt_freq 1000 \
  --num_steps 100000 \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train_and_eval.log"

