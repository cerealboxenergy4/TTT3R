#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import datetime
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from einops import rearrange

# Set random seed for reproducibility.
random.seed(42)

framerate = 30

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--no_vis",
        "--disable_vis",
        action="store_true",
        dest="disable_vis",
        help="Disable point cloud visualization and viewer launch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the point cloud viewer",
    )
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model update type: cut3r or ttt3r",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval for video processing (e.g., 1 means every frame, 2 means every other frame)",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Only used for demo, reset state for extremely long sequence, chunks are aligned via global camera poses",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for the point cloud viewer",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Save state/memory statistics plots and arrays",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="",
        help="Alias used for naming saved stats outputs",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Compute PCA90 component counts for state tokens and LocalMemory",
    )
    parser.add_argument(
        "--single_state_token_idx",
        type=int,
        default=0,
        help="State token index for single-token statistic tracking",
    )
    parser.add_argument(
        "--single_state_stat",
        type=str,
        default="std",
        choices=["std", "l1", "l2"],
        help="Statistic to track for a single state token",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True, reset_interval=10000
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    import roma
    from viser_utils import convert_scene_output_to_glb


    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    if os.path.exists(os.path.join(outdir, "depth")):
        shutil.rmtree(os.path.join(outdir, "depth"))
    if os.path.exists(os.path.join(outdir, "conf")):
        shutil.rmtree(os.path.join(outdir, "conf"))
    if os.path.exists(os.path.join(outdir, "color")):
        shutil.rmtree(os.path.join(outdir, "color"))
    if os.path.exists(os.path.join(outdir, "camera")):
        shutil.rmtree(os.path.join(outdir, "camera"))
    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)
    return pts3ds_other, colors, conf_other, cam_dict

def parse_seq_path(p, frame_interval=1):
    global framerate
    
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = [path for path in all_img_paths 
                    if os.path.splitext(path.lower())[1] in img_extensions]
        
        if not img_paths:
            raise ValueError(f"No image files found in directory {p}")
        
        if frame_interval > 1:
            img_paths = img_paths[::frame_interval]
            print(f" - Image sequence: Total images: {len(all_img_paths)}, "
                  f"Frame interval: {frame_interval}, Images to process: {len(img_paths)}")
        
        framerate = 30.0 / frame_interval
        
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        
        framerate = video_fps / frame_interval
        
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}, Processed Framerate: {framerate}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def compute_state_stats_series(state_args):
    std_values = []
    l1_values = []
    l2_values = []
    for state_arg in state_args:
        state_feat = state_arg[0]
        flat = state_feat.detach().float().reshape(-1)
        if flat.numel() == 0:
            std_values.append(float("nan"))
            l1_values.append(float("nan"))
            l2_values.append(float("nan"))
            continue
        std_values.append(torch.std(flat, unbiased=False).item())
        l1_values.append(torch.mean(torch.abs(flat)).item())
        l2_values.append(torch.sqrt(torch.mean(flat * flat)).item())
    return (
        np.array(std_values, dtype=np.float32),
        np.array(l1_values, dtype=np.float32),
        np.array(l2_values, dtype=np.float32),
    )


def compute_mem_stats_series(state_args):
    std_values = []
    l1_values = []
    l2_values = []
    front_std_values = []
    front_l1_values = []
    front_l2_values = []
    back_std_values = []
    back_l1_values = []
    back_l2_values = []
    for state_arg in state_args:
        mem = state_arg[3]
        flat = mem.detach().float().reshape(-1)
        last_dim = mem.shape[-1]
        half = last_dim // 2
        front_flat = mem[..., :half].detach().float().reshape(-1)
        back_flat = mem[..., half:].detach().float().reshape(-1)
        if flat.numel() == 0:
            std_values.append(float("nan"))
            l1_values.append(float("nan"))
            l2_values.append(float("nan"))
            front_std_values.append(float("nan"))
            front_l1_values.append(float("nan"))
            front_l2_values.append(float("nan"))
            back_std_values.append(float("nan"))
            back_l1_values.append(float("nan"))
            back_l2_values.append(float("nan"))
            continue
        std_values.append(torch.std(flat, unbiased=False).item())
        l1_values.append(torch.norm(flat, p=1).item())
        l2_values.append(torch.norm(flat, p=2).item())
        if front_flat.numel() == 0:
            front_std_values.append(float("nan"))
            front_l1_values.append(float("nan"))
            front_l2_values.append(float("nan"))
        else:
            front_std_values.append(torch.std(front_flat, unbiased=False).item())
            front_l1_values.append(torch.mean(torch.abs(front_flat)).item())
            front_l2_values.append(torch.sqrt(torch.mean(front_flat * front_flat)).item())
        if back_flat.numel() == 0:
            back_std_values.append(float("nan"))
            back_l1_values.append(float("nan"))
            back_l2_values.append(float("nan"))
        else:
            back_std_values.append(torch.std(back_flat, unbiased=False).item())
            back_l1_values.append(torch.mean(torch.abs(back_flat)).item())
            back_l2_values.append(torch.sqrt(torch.mean(back_flat * back_flat)).item())
    return (
        np.array(std_values, dtype=np.float32),
        np.array(l1_values, dtype=np.float32),
        np.array(l2_values, dtype=np.float32),
        np.array(front_std_values, dtype=np.float32),
        np.array(front_l1_values, dtype=np.float32),
        np.array(front_l2_values, dtype=np.float32),
        np.array(back_std_values, dtype=np.float32),
        np.array(back_l1_values, dtype=np.float32),
        np.array(back_l2_values, dtype=np.float32),
    )


def compute_single_state_token_series(state_args, token_idx, stat):
    series = []
    for state_arg in state_args:
        state_feat = state_arg[0]
        if state_feat.ndim < 3 or token_idx < 0 or token_idx >= state_feat.shape[1]:
            series.append(float("nan"))
            continue
        token_vec = state_feat[0, token_idx].detach().float().reshape(-1)
        if token_vec.numel() == 0:
            series.append(float("nan"))
            continue
        if stat == "std":
            series.append(torch.std(token_vec, unbiased=False).item())
        elif stat == "l1":
            series.append(torch.mean(torch.abs(token_vec)).item())
        elif stat == "l2":
            series.append(torch.sqrt(torch.mean(token_vec * token_vec)).item())
        else:
            series.append(float("nan"))
    return np.array(series, dtype=np.float32)


def _svdvals(x):
    try:
        return torch.linalg.svdvals(x)
    except AttributeError:
        return torch.linalg.svd(x, full_matrices=False)[1]


def compute_pca90_components_series(state_args, tensor_index):
    counts = []
    for state_arg in state_args:
        tensor = state_arg[tensor_index]
        if tensor is None or tensor.numel() == 0:
            counts.append(float("nan"))
            continue
        x = tensor.detach().float().reshape(-1, tensor.shape[-1])
        if x.shape[0] < 2 or x.shape[1] < 1:
            counts.append(float("nan"))
            continue
        x = x - x.mean(dim=0, keepdim=True)
        x = x.cpu()
        s = _svdvals(x)
        if x.shape[0] <= 1:
            counts.append(float("nan"))
            continue
        var = (s**2) / (x.shape[0] - 1)
        total = var.sum()
        if total <= 0:
            counts.append(float("nan"))
            continue
        cum = torch.cumsum(var, 0) / total
        k = int((cum < 0.9).sum().item()) + 1
        counts.append(k)
    return np.array(counts, dtype=np.float32)


def save_state_plot(values, plot_path, seq_id, title_suffix, y_label):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    x = np.arange(len(values))
    plt.plot(x, values, marker="o", linewidth=1.5)
    plt.xlabel("Frame index (0 = init state)")
    plt.ylabel(y_label)
    plt.title(f"{title_suffix} over time ({seq_id})")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        reset_interval=args.reset_interval
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path, weights_only=False).to(device)
    model.config.model_update_type = args.model_update_type

    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference_recurrent_lighter(views, model, device)

    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    FPS_num = 1 / per_frame_time
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
    )

    from pathlib import Path

    seq = Path(args.seq_path)
    seq_id = seq.stem if seq.suffix else seq.name
    base_id = args.alias.strip() if args.alias else seq_id
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{base_id}_{timestamp}"

    if args.stats:
        std_dir = os.path.join("experiments", "state_std", run_id)
        std_npy_dir = os.path.join(std_dir, "npy")
        state_std, state_l1, state_l2 = compute_state_stats_series(state_args)
        if len(state_args) > 0:
            entry_count = int(state_args[0][0].numel())
            print(f"State entry count per frame: {entry_count}")
        else:
            print("State entry count per frame: 0 (no state_args)")
        std_path = os.path.join(std_npy_dir, f"{run_id}_state_std.npy")
        l1_path = os.path.join(std_npy_dir, f"{run_id}_state_l1.npy")
        l2_path = os.path.join(std_npy_dir, f"{run_id}_state_l2.npy")
        plot_std_path = os.path.join(std_dir, f"{run_id}_state_std.png")
        plot_l1_path = os.path.join(std_dir, f"{run_id}_state_l1.png")
        plot_l2_path = os.path.join(std_dir, f"{run_id}_state_l2.png")
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(std_npy_dir, exist_ok=True)
        np.save(std_path, state_std)
        np.save(l1_path, state_l1)
        np.save(l2_path, state_l2)
        save_state_plot(state_std, plot_std_path, seq_id, "State token std", "Std")
        save_state_plot(
            state_l1, plot_l1_path, seq_id, "State token L1 mean", "L1 mean"
        )
        save_state_plot(
            state_l2, plot_l2_path, seq_id, "State token L2 mean", "L2 mean"
        )
        print(f"Saved state std series to {std_path}")
        print(f"Saved state l1 series to {l1_path}")
        print(f"Saved state l2 series to {l2_path}")
        print(f"Saved state std plot to {plot_std_path}")
        print(f"Saved state l1 plot to {plot_l1_path}")
        print(f"Saved state l2 plot to {plot_l2_path}")

        pose_dir = os.path.join("experiments", "state_pose", run_id)
        pose_npy_dir = os.path.join(pose_dir, "npy")
        (
            mem_std,
            mem_l1,
            mem_l2,
            mem_front_std,
            mem_front_l1,
            mem_front_l2,
            mem_back_std,
            mem_back_l1,
            mem_back_l2,
        ) = compute_mem_stats_series(state_args)
        if len(state_args) > 0:
            mem_entry_count = int(state_args[0][3].numel())
            print(f"LocalMemory entry count per frame: {mem_entry_count}")
        else:
            print("LocalMemory entry count per frame: 0 (no state_args)")
        mem_std_path = os.path.join(pose_npy_dir, f"{run_id}_mem_std.npy")
        mem_l1_path = os.path.join(pose_npy_dir, f"{run_id}_mem_l1.npy")
        mem_l2_path = os.path.join(pose_npy_dir, f"{run_id}_mem_l2.npy")
        mem_plot_std = os.path.join(pose_dir, f"{run_id}_mem_std.png")
        mem_plot_l1 = os.path.join(pose_dir, f"{run_id}_mem_l1.png")
        mem_plot_l2 = os.path.join(pose_dir, f"{run_id}_mem_l2.png")
        mem_front_std_path = os.path.join(pose_npy_dir, f"{run_id}_mem_front_std.npy")
        mem_front_l1_path = os.path.join(pose_npy_dir, f"{run_id}_mem_front_l1.npy")
        mem_front_l2_path = os.path.join(pose_npy_dir, f"{run_id}_mem_front_l2.npy")
        mem_back_std_path = os.path.join(pose_npy_dir, f"{run_id}_mem_back_std.npy")
        mem_back_l1_path = os.path.join(pose_npy_dir, f"{run_id}_mem_back_l1.npy")
        mem_back_l2_path = os.path.join(pose_npy_dir, f"{run_id}_mem_back_l2.npy")
        mem_plot_front_std = os.path.join(pose_dir, f"{run_id}_mem_front_std.png")
        mem_plot_front_l1 = os.path.join(pose_dir, f"{run_id}_mem_front_l1.png")
        mem_plot_front_l2 = os.path.join(pose_dir, f"{run_id}_mem_front_l2.png")
        mem_plot_back_std = os.path.join(pose_dir, f"{run_id}_mem_back_std.png")
        mem_plot_back_l1 = os.path.join(pose_dir, f"{run_id}_mem_back_l1.png")
        mem_plot_back_l2 = os.path.join(pose_dir, f"{run_id}_mem_back_l2.png")
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(pose_npy_dir, exist_ok=True)
        np.save(mem_std_path, mem_std)
        np.save(mem_l1_path, mem_l1)
        np.save(mem_l2_path, mem_l2)
        np.save(mem_front_std_path, mem_front_std)
        np.save(mem_front_l1_path, mem_front_l1)
        np.save(mem_front_l2_path, mem_front_l2)
        np.save(mem_back_std_path, mem_back_std)
        np.save(mem_back_l1_path, mem_back_l1)
        np.save(mem_back_l2_path, mem_back_l2)
        save_state_plot(mem_std, mem_plot_std, seq_id, "LocalMemory std", "Std")
        save_state_plot(
            mem_l1, mem_plot_l1, seq_id, "LocalMemory L1 mean", "L1 mean"
        )
        save_state_plot(
            mem_l2, mem_plot_l2, seq_id, "LocalMemory L2 mean", "L2 mean"
        )
        save_state_plot(
            mem_front_std,
            mem_plot_front_std,
            seq_id,
            "LocalMemory front-half std",
            "Std",
        )
        save_state_plot(
            mem_front_l1,
            mem_plot_front_l1,
            seq_id,
            "LocalMemory front-half L1 mean",
            "L1 mean",
        )
        save_state_plot(
            mem_front_l2,
            mem_plot_front_l2,
            seq_id,
            "LocalMemory front-half L2 mean",
            "L2 mean",
        )
        save_state_plot(
            mem_back_std,
            mem_plot_back_std,
            seq_id,
            "LocalMemory back-half std",
            "Std",
        )
        save_state_plot(
            mem_back_l1,
            mem_plot_back_l1,
            seq_id,
            "LocalMemory back-half L1 mean",
            "L1 mean",
        )
        save_state_plot(
            mem_back_l2,
            mem_plot_back_l2,
            seq_id,
            "LocalMemory back-half L2 mean",
            "L2 mean",
        )
        print(f"Saved mem std series to {mem_std_path}")
        print(f"Saved mem l1 series to {mem_l1_path}")
        print(f"Saved mem l2 series to {mem_l2_path}")
        print(f"Saved mem front-half std series to {mem_front_std_path}")
        print(f"Saved mem front-half l1 series to {mem_front_l1_path}")
        print(f"Saved mem front-half l2 series to {mem_front_l2_path}")
        print(f"Saved mem back-half std series to {mem_back_std_path}")
        print(f"Saved mem back-half l1 series to {mem_back_l1_path}")
        print(f"Saved mem back-half l2 series to {mem_back_l2_path}")
        print(f"Saved mem std plot to {mem_plot_std}")
        print(f"Saved mem l1 plot to {mem_plot_l1}")
        print(f"Saved mem l2 plot to {mem_plot_l2}")
        print(f"Saved mem front-half std plot to {mem_plot_front_std}")
        print(f"Saved mem front-half l1 plot to {mem_plot_front_l1}")
        print(f"Saved mem front-half l2 plot to {mem_plot_front_l2}")
        print(f"Saved mem back-half std plot to {mem_plot_back_std}")
        print(f"Saved mem back-half l1 plot to {mem_plot_back_l1}")
        print(f"Saved mem back-half l2 plot to {mem_plot_back_l2}")

        single_series = compute_single_state_token_series(
            state_args, args.single_state_token_idx, args.single_state_stat
        )
        single_label = f"State token {args.single_state_token_idx} {args.single_state_stat}"
        single_base = (
            f"{run_id}_state_token{args.single_state_token_idx}_{args.single_state_stat}"
        )
        single_npy_path = os.path.join(std_npy_dir, f"{single_base}.npy")
        single_plot_path = os.path.join(std_dir, f"{single_base}.png")
        np.save(single_npy_path, single_series)
        save_state_plot(
            single_series,
            single_plot_path,
            seq_id,
            single_label,
            args.single_state_stat,
        )
        print(f"Saved single-token series to {single_npy_path}")
        print(f"Saved single-token plot to {single_plot_path}")

    if args.pca:
        pca_dir = os.path.join("experiments", "state_pca", run_id)
        pca_npy_dir = os.path.join(pca_dir, "npy")
        os.makedirs(pca_dir, exist_ok=True)
        os.makedirs(pca_npy_dir, exist_ok=True)
        state_pca90 = compute_pca90_components_series(state_args, 0)
        mem_pca90 = compute_pca90_components_series(state_args, 3)
        state_pca_npy = os.path.join(pca_npy_dir, f"{run_id}_state_pca90.npy")
        mem_pca_npy = os.path.join(pca_npy_dir, f"{run_id}_mem_pca90.npy")
        state_pca_plot = os.path.join(pca_dir, f"{run_id}_state_pca90.png")
        mem_pca_plot = os.path.join(pca_dir, f"{run_id}_mem_pca90.png")
        np.save(state_pca_npy, state_pca90)
        np.save(mem_pca_npy, mem_pca90)
        save_state_plot(
            state_pca90,
            state_pca_plot,
            seq_id,
            "State PCA90 components",
            "Components",
        )
        save_state_plot(
            mem_pca90,
            mem_pca_plot,
            seq_id,
            "LocalMemory PCA90 components",
            "Components",
        )
        print(f"Saved state PCA90 series to {state_pca_npy}")
        print(f"Saved LocalMemory PCA90 series to {mem_pca_npy}")
        print(f"Saved state PCA90 plot to {state_pca_plot}")
        print(f"Saved LocalMemory PCA90 plot to {mem_pca_plot}")

    if args.disable_vis:
        print("Visualization disabled; skipping point cloud viewer.")
        return

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    from viser_utils import PointCloudViewer

    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size,
        port = args.port,
        downsample_factor=args.downsample_factor
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
