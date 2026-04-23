# Datasets



## Optical Flow

The datasets used to train and evaluate our GMFlow model are as follows:

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Sintel](http://sintel.is.tue.mpg.de/)
- [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

By default the dataloader [dataloader/flow/datasets.py](dataloader/flow/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/flow/datasets.py](dataloader/flow/datasets.py).



## Stereo Matching

The datasets used to train and evaluate our GMStereo model are as follows:

- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [TartanAir](https://github.com/castacks/tartanair_tools)
- [Falling Things](https://research.nvidia.com/publication/2018-06_Falling-Things)
- [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)
- [CREStereo Dataset](https://github.com/megvii-research/CREStereo/blob/master/dataset_download.sh)
- [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
- [Middlebury](https://vision.middlebury.edu/stereo/data/)
- [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
- [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)

By default the dataloader [dataloader/stereo/datasets.py](dataloader/stereo/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/stereo/datasets.py](dataloader/flow/datasets.py).

### Cloud Stereo metadata format

You can also train stereo with `--stage cloud_stereo` using metadata JSON files (for example from [jacoblin/cloud-stereo](https://huggingface.co/datasets/jacoblin/cloud-stereo)).
Each JSON file should include:

- `h`: image height used for rendering
- `frames`: list of entries with `left_image_path`, `right_image_path`, `disparity_path`
- Optional (global or per-frame) calibration fields for metric-depth evaluation:
  - `focal_length_px` (or `focal_px` / `fx`)
  - `baseline_m` (or `baseline`)
  - `camera_height_m` (or `camera_height`) for height RMSE
  - If `focal_length_px` is not present, `fov_x_y` + `h` are also supported to derive focal length.

Use CLI flags to point to your converted metadata files:

```
python main_stereo.py \
  --stage cloud_stereo \
  --cloudstereo_root datasets/cloud-stereo \
  --cloudstereo_train_json train.json
```

A ready-to-edit training launcher is provided at [scripts/gmstereo_scale2_cloud_stereo_train.sh](scripts/gmstereo_scale2_cloud_stereo_train.sh).

#### Downloading from Hugging Face

To prepare the [jacoblin/cloud-stereo](https://huggingface.co/datasets/jacoblin/cloud-stereo)
dataset directly into a UniMatch-compatible layout:

```bash
pip install huggingface_hub
python tools/prepare_cloudstereo_hf.py \
  --repo_id jacoblin/cloud-stereo \
  --output_dir datasets/cloud-stereo
```

This generates `train.json` and `val.json` under `datasets/cloud-stereo/` and
creates a `datasets/cloud-stereo/data` symlink to the downloaded snapshot.

#### Reproducing the BMVC 2025 Cloud-Stereo fine-tuning protocol

The Cloud-Stereo paper reports gains from fine-tuning a learned stereo model on
their synthetic split and evaluating on real-world data. In this repo, the
closest protocol is:

1) fine-tune GMStereo with `--stage cloud_stereo` using `train.json`, then  
2) run `scripts/gmstereo_cloudstereo_evaluate.sh` with `--val_dataset cloudstereo`.

A one-command launcher is available at
[scripts/gmstereo_cloudstereo_reproduce_bmvc2025.sh](scripts/gmstereo_cloudstereo_reproduce_bmvc2025.sh).




## Depth Estimation

The datasets used to train and evaluate our GMDepth model are as follows:

- [DeMoN](https://github.com/lmb-freiburg/demon)
- [ScanNet](http://www.scan-net.org/)

We support downloading and extracting the DeMoN dataset in our code: [dataloader/depth/download_demon_train.sh](dataloader/depth/download_demon_train.sh),  [dataloader/depth/download_demon_test.sh](dataloader/depth/download_demon_test.sh),  [dataloader/depth/prepare_demon_train.sh](dataloader/depth/prepare_demon_train.sh) and  [dataloader/depth/prepare_demon_test.sh](dataloader/depth/prepare_demon_test.sh).

By default the dataloader [dataloader/depth/datasets.py](dataloader/depth/datasets.py) assumes the datasets are located in the `datasets` directory.

It is recommended to symlink your dataset root to `datasets`:

```
ln -s $YOUR_DATASET_ROOT datasets
```

Otherwise, you may need to change the corresponding paths in [dataloader/depth/datasets.py](dataloader/depth/datasets.py).

