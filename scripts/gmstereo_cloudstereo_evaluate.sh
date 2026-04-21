#!/usr/bin/env bash

# Evaluate a trained GMStereo checkpoint on CloudStereo validation split.
# Update paths for your setup before running.
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--eval \
--resume checkpoints_stereo/cloudstereo-gmstereo/step_100000.pth \
--val_dataset cloudstereo \
--stage cloud_stereo \
--cloudstereo_root datasets/cloud-stereo \
--cloudstereo_val_json val.json \
--num_scales 2 \
--upsample_factor 4 \
--reg_refine \
--num_reg_refine 3
