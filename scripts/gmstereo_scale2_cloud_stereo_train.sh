#!/usr/bin/env bash
set -euo pipefail

# GMStereo scale-2 fine-tuning on Cloud Stereo metadata format
# Defaults mirror the current single-GPU workflow.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints_stereo/cloud-stereo-gmstereo-scale2}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-pretrained/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth}
CLOUDSTEREO_ROOT=${CLOUDSTEREO_ROOT:-/nfsscratch/wndrsn/cloud_stereo}
CLOUDSTEREO_TRAIN_JSON=${CLOUDSTEREO_TRAIN_JSON:-synthetic_dataset/train.json}
CLOUDSTEREO_VAL_JSON=${CLOUDSTEREO_VAL_JSON:-synthetic_dataset/test.json}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_STEPS=${NUM_STEPS:-100000}

if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "Missing pretrained checkpoint: ${PRETRAINED_CKPT}"
  echo "Set PRETRAINED_CKPT=/path/to/ckpt.pth and re-run."
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python main_stereo.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --resume "${PRETRAINED_CKPT}" \
  --no_resume_optimizer \
  --stage cloud_stereo \
  --cloudstereo_root "${CLOUDSTEREO_ROOT}" \
  --cloudstereo_train_json "${CLOUDSTEREO_TRAIN_JSON}" \
  --cloudstereo_val_json "${CLOUDSTEREO_VAL_JSON}" \
  --batch_size "${BATCH_SIZE}" \
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
  --summary_freq 100 \
  --val_freq 1000 \
  --save_ckpt_freq 1000 \
  --save_latest_ckpt_freq 1000 \
  --num_steps "${NUM_STEPS}" \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
