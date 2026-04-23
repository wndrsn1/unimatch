#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
  --checkpoint_dir checkpoints_stereo/cloud-stereo-gmstereo-scale2 \
  --resume pretrained/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth \
  --no_resume_optimizer \
  --stage cloud_stereo \
  --cloudstereo_root /nfsscratch/wndrsn/cloud_stereo \
  --cloudstereo_train_json synthetic_dataset/train.json \
  --cloudstereo_val_json synthetic_dataset/test.json \
  --batch_size 4 \
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
  --num_steps 100000
