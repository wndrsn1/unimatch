#!/usr/bin/env bash

# Evaluate a trained GMStereo checkpoint on CloudStereo validation split.
# Prints cloudstereo_epe/d1 and depth-style stats:
# abs_rel_med, sq_rel_med, rmse_med, rmse_log_med, delta1_mean.
# Also prints disparity RMSE, and (if metadata includes calibration) depth/height RMSE.
# Update paths for your setup before running.
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--eval \
--resume checkpoints_stereo/cloudstereo-gmstereo/step_100000.pth \
--val_dataset cloudstereo \
--stage cloud_stereo \
--cloudstereo_root datasets/cloud-stereo \
--cloudstereo_val_json val.json \
--save_vis_disp \
--save_dir output/cloudstereo_eval_panels \
--num_scales 2 \
--upsample_factor 4 \
--reg_refine \
--num_reg_refine 3
