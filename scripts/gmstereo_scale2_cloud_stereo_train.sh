#!/usr/bin/env bash

# GMStereo scale-2 retraining on Cloud Stereo metadata format

NUM_GPUS=8
DATA_ROOT=datasets/cloud-stereo
TRAIN_JSONS=(train.json)
CHECKPOINT_DIR=checkpoints_stereo/cloud-stereo-gmstereo-scale2
<<<<<<< codex/retrain-gmstereo-scale2-sceneflow-model-cyi1y3
PRETRAINED_DIR=pretrained
PRETRAINED_CKPT=${PRETRAINED_DIR}/gmflow-scale2-things-36579974.pth

if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "Missing pretrained checkpoint: ${PRETRAINED_CKPT}"
  echo "Download it and place it at that path (or edit PRETRAINED_CKPT in this script)."
  exit 1
fi
=======
>>>>>>> master

mkdir -p "${CHECKPOINT_DIR}" && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_stereo.py \
--launcher pytorch \
--checkpoint_dir "${CHECKPOINT_DIR}" \
<<<<<<< codex/retrain-gmstereo-scale2-sceneflow-model-cyi1y3
--resume "${PRETRAINED_CKPT}" \
=======
--resume pretrained/gmflow-scale2-things-36579974.pth \
>>>>>>> master
--no_resume_optimizer \
--stage cloud_stereo \
--cloudstereo_root "${DATA_ROOT}" \
--cloudstereo_train_json "${TRAIN_JSONS[@]}" \
--batch_size 32 \
--val_dataset things kitti15 \
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
--val_freq 10000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
