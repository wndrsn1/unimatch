import argparse
import glob
import json
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from unimatch.unimatch import UniMatch
from utils.utils import InputPadder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GMStereoPredictor:
    def __init__(self, ckpt_path, device='cuda',
                 num_scales=2, upsample_factor=4, reg_refine=False, num_reg_refine=1,
                 attn_type='self_swin2d_cross_1d',
                 attn_splits_list=(2, 8), corr_radius_list=(-1, 4), prop_radius_list=(-1, 1),
                 padding_factor=32):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.padding_factor = padding_factor
        self.attn_type = attn_type
        self.attn_splits_list = list(attn_splits_list)
        self.corr_radius_list = list(corr_radius_list)
        self.prop_radius_list = list(prop_radius_list)
        self.num_reg_refine = num_reg_refine

        self.model = UniMatch(num_scales=num_scales,
                              upsample_factor=upsample_factor,
                              reg_refine=reg_refine,
                              task='stereo').to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.eval()

    def _to_tensor(self, rgb):
        x = rgb.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN)[None, None, :]) / np.array(IMAGENET_STD)[None, None, :]
        x = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def predict(self, left_rgb, right_rgb):
        left = self._to_tensor(left_rgb)
        right = self._to_tensor(right_rgb)
        padder = InputPadder(left.shape, padding_factor=self.padding_factor)
        left, right = padder.pad(left, right)

        pred = self.model(left, right,
                          attn_type=self.attn_type,
                          attn_splits_list=self.attn_splits_list,
                          corr_radius_list=self.corr_radius_list,
                          prop_radius_list=self.prop_radius_list,
                          num_reg_refine=self.num_reg_refine,
                          task='stereo')['flow_preds'][-1]
        pred = padder.unpad(pred)[0].detach().cpu().numpy()
        return pred


class LidarData:
    def __init__(self, data=None, data_locs=None, num_gates=0):
        self.data = data
        self.data_locs = data_locs
        self.num_gates = num_gates

    @classmethod
    def fromfile(cls, filename):
        with open(filename) as f:
            header = [f.readline().split(':', maxsplit=1) for _ in range(17)]
            num_gates = int(header[2][1].strip())
            data = []
            data_locs = []
            while True:
                line = f.readline()
                if not line:
                    break
                parts = line.split()
                if len(parts) == 0:
                    break
                data_locs.append(np.array(parts, dtype=float))
                ray_data = np.array([f.readline().split() for _ in range(num_gates)], dtype=float)
                data.append(ray_data)
        return cls(data=np.array(data), data_locs=np.array(data_locs), num_gates=num_gates)

    def getBackscatter(self):
        return self.data[:, 20:, 2]


def project_lidar_to_right(lidar_to_right_cam, right_intrinsic, azi, elev, depth):
    azi = -azi + 270.0
    x = depth * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azi))
    z = depth * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azi))
    y = -depth * np.sin(np.deg2rad(elev))
    lidar_xyz = np.array([x, y, z, 1.0], dtype=np.float64)
    cam_xyz = lidar_to_right_cam @ lidar_xyz
    if cam_xyz[2] <= 0:
        return None, None
    proj = right_intrinsic @ (cam_xyz[:3] / cam_xyz[2])
    return proj[:2], np.linalg.norm(cam_xyz[:3])


def find_cloud_in_backscatter(backscatter, dist_thresh=120, smooth_window=9, sigma_thresh=2.0):
    b = np.asarray(backscatter, dtype=np.float32)
    if b.ndim != 1 or len(b) <= dist_thresh:
        return None, None
    k = np.ones(smooth_window, dtype=np.float32) / smooth_window
    bs = np.convolve(b, k, mode="same")
    search = bs[dist_thresh:]
    thresh = np.median(search) + sigma_thresh * max(np.std(search), 1e-6)
    idx = np.where(search > thresh)[0]
    if len(idx) == 0:
        return None, None
    cloud_idx = int(idx[0] + dist_thresh)
    cloud_depth = (cloud_idx + 20) * 3.0
    return cloud_idx, cloud_depth


def load_lidar_data_right(lidar_files, hour, lidar_to_right, right_intrinsic, max_frames=717):
    decimal_time, azi, elev, backscatter = [], [], [], []
    for lidar_file in lidar_files:
        ld = LidarData.fromfile(lidar_file)
        backscatter.extend(ld.getBackscatter())
        decimal_time.extend(ld.data_locs[:, 0].tolist())
        azi.extend(ld.data_locs[:, 1].tolist())
        elev.extend(ld.data_locs[:, 2].tolist())

    decimal_time = np.array(decimal_time, dtype=np.float64)
    azi = np.array(azi, dtype=np.float64)
    elev = np.array(elev, dtype=np.float64)

    lidar_output = defaultdict(list)
    frame_times = hour + (10.0 + 5.0 * np.arange(max_frames)) / 3600.0
    for frame_idx, camera_time in enumerate(frame_times):
        idxs = np.where(np.abs(decimal_time - camera_time) < 2.5 / 3600.0)[0]
        for i in idxs:
            cloud_idx, cloud_depth = find_cloud_in_backscatter(backscatter[i])
            if cloud_depth is None:
                continue
            uv_right, right_cam_depth = project_lidar_to_right(
                lidar_to_right, right_intrinsic, azi[i], elev[i], cloud_depth
            )
            if uv_right is None:
                continue
            lidar_output[frame_idx].append({
                "right_cam_depth": float(right_cam_depth),
                "right_cam_xy": np.array(uv_right, dtype=np.float32),
                "cloud_idx": int(cloud_idx),
            })
    return lidar_output


def load_images(file_dir, date, hour, frame_idx, meta):
    h, w = meta["h"], meta["w"]
    left_video = cv2.VideoCapture(f"{file_dir}/left_images/{date}/{date}_{hour:0>2}.mp4")
    right_video = cv2.VideoCapture(f"{file_dir}/right_images/{date}/{date}_{hour:0>2}.mp4")
    left_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    right_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    okL, left_image = left_video.read()
    okR, right_image = right_video.read()
    left_video.release()
    right_video.release()
    if not okL or left_image is None or not okR or right_image is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    left_image = cv2.cvtColor(cv2.resize(left_image, (w, h)), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.resize(right_image, (w, h)), cv2.COLOR_BGR2RGB)
    return left_image, right_image


def bilinear_sample_scalar(img, x, y):
    h, w = img.shape[:2]
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0
    v00, v10, v01, v11 = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    return float(v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy)


def pick_feature_near_lidar_point(image_gray, x, y, roi_half=15, max_corners=20):
    h, w = image_gray.shape[:2]
    x0, x1 = max(0, int(round(x)) - roi_half), min(w, int(round(x)) + roi_half + 1)
    y0, y1 = max(0, int(round(y)) - roi_half), min(h, int(round(y)) + roi_half + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = image_gray[y0:y1, x0:x1]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=0.01, minDistance=3, blockSize=5)
    if pts is None:
        return None
    pts = pts.reshape(-1, 2)
    pts[:, 0] += x0
    pts[:, 1] += y0
    target = np.array([x, y], dtype=np.float32)
    return pts[np.argmin(np.sum((pts - target[None, :]) ** 2, axis=1))]


def estimate_disparities_from_right_seed_points(left_image, right_image, lidar_points, fx, baseline_m, predictor):
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
    disparity_map = predictor.predict(left_image, right_image)
    results = []
    for p in lidar_points:
        if "right_cam_xy" not in p:
            continue
        x_right_seed, y_right_seed = p["right_cam_xy"]
        gt_depth = p["right_cam_depth"]
        feat = pick_feature_near_lidar_point(right_gray, x_right_seed, y_right_seed, roi_half=15)
        if feat is None:
            continue
        xr, yr = float(feat[0]), float(feat[1])
        d0 = bilinear_sample_scalar(disparity_map, xr, yr)
        if d0 is None or d0 <= 0.25:
            continue
        xl = xr + d0
        d1 = bilinear_sample_scalar(disparity_map, xl, yr)
        if d1 is None or d1 <= 0.25:
            continue
        disparity = d1
        pred_depth = fx * baseline_m / disparity
        results.append({
            "disparity": float(disparity),
            "pred_depth": float(pred_depth),
            "gt_depth": float(gt_depth),
        })
    return results, disparity_map


def summarize_results(results):
    if len(results) == 0:
        return None
    gt = np.array([r["gt_depth"] for r in results], dtype=np.float32)
    pred = np.array([r["pred_depth"] for r in results], dtype=np.float32)
    err = pred - gt
    return {"n": len(results), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}


def plot_disparity(disparity_map, frame_idx):
    plt.figure(figsize=(8, 5))
    plt.imshow(disparity_map, cmap="plasma")
    plt.colorbar(label="Disparity (px)")
    plt.title(f"GMStereo disparity map (frame {frame_idx})")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--hour', required=True, type=int)
    parser.add_argument('--frame_start', default=80, type=int)
    parser.add_argument('--frame_end', default=100, type=int)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--baseline_m', default=62.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    with open(f"{args.file_dir}/calib.json", "r") as fp:
        meta = json.load(fp)
    right_intrinsic = np.array(meta["right_intrinsic"], dtype=np.float64)
    lidar_to_right = np.array(meta["lidar_to_right_cam"], dtype=np.float64)
    fx = float(right_intrinsic[0, 0])

    predictor = GMStereoPredictor(args.checkpoint, device=args.device)
    lidar_files = sorted(glob.glob(f"{args.file_dir}/lidar/{args.date}/{args.date}_{args.hour:0>2}*.hpl"))
    lidar_data = load_lidar_data_right(lidar_files, args.hour, lidar_to_right, right_intrinsic)

    for frame_idx in range(args.frame_start, args.frame_end):
        try:
            left_image, right_image = load_images(args.file_dir, args.date, args.hour, frame_idx, meta)
            lidar_points = lidar_data[frame_idx]
            results, disparity_map = estimate_disparities_from_right_seed_points(
                left_image, right_image, lidar_points, fx, args.baseline_m, predictor
            )
            summary = summarize_results(results)
            print(f"frame {frame_idx}: lidar points={len(lidar_points)}, matched features={len(results)}")
            if summary is not None:
                print(f"RMSE={summary['rmse']:.3f} m, MAE={summary['mae']:.3f} m, Bias={summary['bias']:.3f} m")
            else:
                print("No valid matches found.")
            plot_disparity(disparity_map, frame_idx)
        except Exception as e:
            print(f"Skipping frame {frame_idx}: {e}")


if __name__ == "__main__":
    main()
