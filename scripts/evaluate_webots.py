import argparse
import csv
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# ============================================================
# Webot Camera Calibration Matrix
# ============================================================

Webot_K = np.array([
    [585.75607095, 0.0,          319.5],
    [0.0,          585.75607095, 239.5],
    [0.0,          0.0,            1.0]
])

# ============================================================
# Webots folder helpers
# ============================================================

def get_webots_samples(data_dir):
    data_dir = Path(data_dir)
    rgb_dir = data_dir / "rgb"
    depth_dir = data_dir / "depth"

    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'rgb' and 'depth' folders under {data_dir.resolve()}"
        )

    rgb_files = {
        path.stem.removeprefix("rgb_"): path
        for path in rgb_dir.glob("rgb_*.png")
    }
    depth_files = {
        path.stem.removeprefix("depth_"): path
        for path in depth_dir.glob("depth_*.npy")
    }
    keys = sorted(rgb_files.keys() & depth_files.keys())

    if not keys:
        raise FileNotFoundError(
            f"No matching rgb_###.png/depth_###.npy pairs found in {data_dir.resolve()}"
        )

    missing_depth = sorted(rgb_files.keys() - depth_files.keys())
    missing_rgb = sorted(depth_files.keys() - rgb_files.keys())
    if missing_depth or missing_rgb:
        raise RuntimeError(
            "Webots RGB/depth files are not paired: "
            f"missing depth={missing_depth[:3]}, missing rgb={missing_rgb[:3]}"
        )

    return [(key, rgb_files[key], depth_files[key]) for key in keys]


def inspect_webot_file(data_dir):
    data_dir = Path(data_dir)
    samples = get_webots_samples(data_dir)
    key, rgb_path, depth_path = samples[0]
    rgb = read_rgb(rgb_path)
    depth = read_depth(depth_path)
    visualized_dir = data_dir / "visualized"
    visualized_count = len(
        list(visualized_dir.glob("comparison_*.png"))
    ) if visualized_dir.is_dir() else 0
    print(f"Webots samples: {len(samples)}")
    print(f"First sample  : {key}")
    print(f"RGB           : {rgb.shape} {rgb.dtype}")
    print(f"Depth         : {depth.shape} {depth.dtype} (meters)")
    print(f"Visualized    : {visualized_count} comparison image(s)")


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read RGB image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def read_depth(path):
    depth = np.asarray(np.load(path), dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"Expected 2D depth array, got {depth.shape}: {path}")
    if not np.isfinite(depth).any():
        raise RuntimeError(f"Depth array contains no finite values: {path}")
    return np.ascontiguousarray(depth)

# ============================================================
# Sample selection
# ============================================================
def choose_indices(
    total_samples,
    num_samples,
    seed,
):
    """
    num_samples <= 0 -> evaluate every labeled frame.
    Otherwise choose a deterministic random subset.
    """
    all_indices = np.arange(
        total_samples,
        dtype=np.int64,
    )

    if (
        num_samples <= 0
        or num_samples >= total_samples
    ):
        return all_indices

    rng = np.random.default_rng(
        seed
    )

    chosen = rng.choice(
        all_indices,
        size=num_samples,
        replace=False,
    )

    return np.sort(chosen)


# ============================================================
# Optional NYU evaluation crop
# ============================================================
def make_spatial_mask(
    height,
    width,
    crop_mode,
):
    mask = np.ones(
        (height, width),
        dtype=bool,
    )

    if crop_mode == "none":
        return mask

    if crop_mode == "nyu":
        # Common NYUv2 evaluation crop:
        # y: 45..470, x: 41..600 for 480x640.
        # Scale coordinates if input resolution differs.
        y1 = int(
            round(
                45 * height / 480.0
            )
        )
        y2 = int(
            round(
                471 * height / 480.0
            )
        )

        x1 = int(
            round(
                41 * width / 640.0
            )
        )
        x2 = int(
            round(
                601 * width / 640.0
            )
        )

        mask[:] = False
        mask[
            y1:y2,
            x1:x2,
        ] = True

        return mask

    raise ValueError(
        f"Unknown crop mode: {crop_mode}"
    )


# ============================================================
# Metrics
# ============================================================
def compute_metrics(
    pred_depth,
    gt_depth,
    valid_mask,
):
    valid = (
        valid_mask
        & np.isfinite(pred_depth)
        & (pred_depth > 0)
        & np.isfinite(gt_depth)
        & (gt_depth > 0)
    )

    count = int(
        valid.sum()
    )

    if count == 0:
        return None

    pred = pred_depth[
        valid
    ].astype(np.float64)

    gt = gt_depth[
        valid
    ].astype(np.float64)

    abs_error = np.abs(
        pred - gt
    )

    ratio = np.maximum(
        pred / gt,
        gt / pred,
    )

    return {
        "valid_pixels": count,

        "mae_m": float(
            np.mean(abs_error)
        ),

        "mae_cm": float(
            np.mean(abs_error)
            * 100.0
        ),

        "absrel_pct": float(
            np.mean(
                abs_error / gt
            )
            * 100.0
        ),

        "delta1_pct": float(
            np.mean(
                ratio < 1.25
            )
            * 100.0
        ),

        # Used only to construct exact pooled metrics.
        "_pred": pred,
        "_gt": gt,
    }


def make_range_masks(
    gt_depth,
    spatial_mask,
):
    base = (
        spatial_mask
        & np.isfinite(gt_depth)
        & (gt_depth >= 0.1)
    )

    return {
        "0-3m": (
            base
            & (gt_depth <= 3.0)
        ),

        "0-6m": (
            base
            & (gt_depth <= 6.0)
        ),

        "0-10m": (
            base
            & (gt_depth <= 10.0)
        ),
    }


# ============================================================
# Point-cloud cache export
# ============================================================
def build_corresponding_point_clouds(
    rgb,
    gt_depth,
    pred_depth,
    K,
    spatial_mask,
    stride=4,
    max_depth=10.0,
):
    """
    Build GT and prediction point clouds from exactly the same image pixels.

    The clouds are reconstructed in a camera-centric coordinate system.
    Y is flipped only for a more natural upright visualization.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    valid = (
        spatial_mask
        & np.isfinite(gt_depth)
        & (gt_depth >= 0.1)
        & (gt_depth <= max_depth)
        & np.isfinite(pred_depth)
        & (pred_depth > 0)
    )

    sampled = np.zeros_like(valid, dtype=bool)
    sampled[::stride, ::stride] = True
    valid &= sampled

    v, u = np.nonzero(valid)

    if u.size == 0:
        return None

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z_gt = gt_depth[valid].astype(np.float32)
    z_pred = pred_depth[valid].astype(np.float32)

    x_gt = ((u.astype(np.float32) - cx) * z_gt / fx)
    y_gt = -((v.astype(np.float32) - cy) * z_gt / fy)

    x_pred = ((u.astype(np.float32) - cx) * z_pred / fx)
    y_pred = -((v.astype(np.float32) - cy) * z_pred / fy)

    gt_points = np.column_stack((x_gt, y_gt, z_gt)).astype(np.float32)
    pred_points = np.column_stack((x_pred, y_pred, z_pred)).astype(np.float32)

    colors = rgb[valid].astype(np.uint8)
    abs_error_m = np.abs(z_pred - z_gt).astype(np.float32)
    rel_error = (abs_error_m / np.maximum(z_gt, 1e-6)).astype(np.float32)

    return {
        "gt_points": gt_points,
        "pred_points": pred_points,
        "colors": colors,
        "abs_error_m": abs_error_m,
        "rel_error": rel_error,
        "u": u.astype(np.int16),
        "v": v.astype(np.int16),
    }


def save_pointcloud_cache(
    out_dir,
    dataset_index,
    rgb,
    gt_depth,
    pred_depth,
    K,
    spatial_mask,
    stride,
):
    pc = build_corresponding_point_clouds(
        rgb=rgb,
        gt_depth=gt_depth,
        pred_depth=pred_depth,
        K=K,
        spatial_mask=spatial_mask,
        stride=stride,
        max_depth=10.0,
    )

    if pc is None:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / f"frame_{dataset_index:04d}.npz",
        dataset_index=np.int32(dataset_index),
        K=K.astype(np.float32),
        **pc,
    )


# ============================================================
# DA3
# ============================================================
def infer_da3(
    model,
    image_path,
    K_original,
    original_width,
    original_height,
    process_res,
):
    K_batch = (
        K_original[None, ...]
        .astype(np.float32)
    )

    # Reset peak memory before the timed inference.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model.inference(
            image=[str(image_path)],
            intrinsics=K_batch,
            process_res=process_res,
            process_res_method=(
                "upper_bound_resize"
            ),
            export_dir=None,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    fps = 1.0 / elapsed

    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    raw_depth = np.asarray(
        prediction.depth[0],
        dtype=np.float32,
    )

    processed_h, processed_w = (
        raw_depth.shape
    )

    returned_intrinsics = getattr(
        prediction,
        "intrinsics",
        None,
    )

    if returned_intrinsics is not None:
        K_processed = np.asarray(
            returned_intrinsics[0],
            dtype=np.float32,
        )

        fx_processed = float(
            K_processed[0, 0]
        )

        fy_processed = float(
            K_processed[1, 1]
        )

    else:
        # Fallback for a pure resize.
        fx_processed = (
            float(K_original[0, 0])
            * processed_w
            / original_width
        )

        fy_processed = (
            float(K_original[1, 1])
            * processed_h
            / original_height
        )

    focal_processed = (
        0.5
        * (
            fx_processed
            + fy_processed
        )
    )

    # Same DA3METRIC-LARGE metric conversion
    # used in the KITTI evaluator.
    metric_depth_processed = (
        focal_processed
        * raw_depth
        / 300.0
    ).astype(np.float32)

    metric_depth = cv2.resize(
        metric_depth_processed,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    return {
        "depth_m": metric_depth,
        "fps": fps,
        "peak_vram_gb": (
            peak_vram_gb
        ),
        "processed_width": (
            processed_w
        ),
        "processed_height": (
            processed_h
        ),
    }





# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DA3METRIC-LARGE"
            "on Webot Simulation Images"
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="Webots-Data",
        help="Folder containing rgb/rgb_###.png and depth/depth_###.npy.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "depth-anything/"
            "DA3METRIC-LARGE"
        ),
    )

    parser.add_argument(
        "--process_res",
        type=int,
        default=1344,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help=(
            "Number of deterministic samples. "
            "Use 0 for all labeled frames."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--crop",
        choices=[
            "none",
            "nyu",
        ],
        default="none",
        help=(
            "'none' evaluates all valid pixels. "
            "'nyu' applies the common NYUv2 crop."
        ),
    )

    parser.add_argument(
        "--per_image_csv",
        type=str,
        default=(
            "webots_da3_metric_per_image.csv"
        ),
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        default=(
            "webots_da3_metric_summary.csv"
        ),
    )

    parser.add_argument(
        "--pc_cache_dir",
        type=str,
        default="webots_da3_pc_cache",
        help=(
            "Directory for interactive point-cloud viewer cache. "
            "Use an empty string to disable export."
        ),
    )

    parser.add_argument(
        "--pc_stride",
        type=int,
        default=4,
        help=(
            "Spatial sampling stride for point-cloud export. "
            "4 gives about 19k points for a full 480x640 frame."
        ),
    )

    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help=(
            "Print Webots folder structure "
            "and exit."
        ),
    )

    args = parser.parse_args()

    samples = get_webots_samples(args.data_dir)

    print("=" * 80)
    print(
        "DEPTH ANYTHING 3 - "
        "WEBOTS METRIC DEPTH EVALUATION"
    )
    print("=" * 80)

    if args.inspect_only:
        inspect_webot_file(args.data_dir)
        return

    total_samples = len(samples)

    # Keep the existing evaluation block scoped without an HDF5 context.
    with nullcontext():

        indices = choose_indices(
            total_samples,
            args.num_samples,
            args.seed,
        )

        print()
        print(
            f"Total labeled frames : "
            f"{total_samples}"
        )

        print(
            f"Evaluating frames    : "
            f"{len(indices)}"
        )

        print(
            f"Sampling seed        : "
            f"{args.seed}"
        )

        print(
            f"Process resolution   : "
            f"{args.process_res}"
        )

        print(
            f"Crop                 : "
            f"{args.crop}"
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device               : "
            f"{device}"
        )

        if device.type == "cuda":
            print(
                f"GPU                  : "
                f"{torch.cuda.get_device_name(0)}"
            )

        print()
        print(
            "Loading DA3METRIC-LARGE..."
        )

        model = (
            DepthAnything3
            .from_pretrained(
                args.model
            )
            .to(device)
        )

        model.eval()

        print("Model loaded.")
        print()

        rows = []

        pooled = {
            "0-3m": [],
            "0-6m": [],
            "0-10m": [],
        }

        fps_values = []
        vram_values = []

        with tempfile.TemporaryDirectory(
            prefix="webots_da3_"
        ) as tmp_dir:

            temp_image_path = (
                Path(tmp_dir)
                / "frame.png"
            )

            # One untimed warm-up frame.
            warmup_index = int(
                indices[0]
            )

            warmup_rgb = read_rgb(samples[warmup_index][1])

            warmup_h, warmup_w = (
                warmup_rgb.shape[:2]
            )

            if (
                warmup_w,
                warmup_h,
            ) != (640, 480):
                print(
                    "[WARN] NYUv2 frame is not "
                    "640x480. Intrinsics will be "
                    "scaled from the canonical "
                    "640x480 calibration."
                )

            warmup_K = Webot_K.copy()

            warmup_K[0, :] *= (
                warmup_w / 640.0
            )

            warmup_K[1, :] *= (
                warmup_h / 480.0
            )

            cv2.imwrite(
                str(temp_image_path),
                cv2.cvtColor(
                    warmup_rgb,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            with torch.inference_mode():
                _ = model.inference(
                    image=[
                        str(temp_image_path)
                    ],
                    intrinsics=(
                        warmup_K[
                            None,
                            ...
                        ]
                    ),
                    process_res=(
                        args.process_res
                    ),
                    process_res_method=(
                        "upper_bound_resize"
                    ),
                    export_dir=None,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for position, dataset_index in enumerate(
                indices,
                1,
            ):
                dataset_index = int(
                    dataset_index
                )

                sample_key, rgb_path, depth_path = samples[dataset_index]
                dataset_index = int(sample_key)
                rgb = read_rgb(rgb_path)
                gt_depth = read_depth(depth_path)

                height, width = (
                    rgb.shape[:2]
                )

                if gt_depth.shape != (
                    height,
                    width,
                ):
                    raise RuntimeError(
                        f"Frame {dataset_index}: "
                        f"RGB={rgb.shape}, "
                        f"depth={gt_depth.shape}"
                    )

                # Scale canonical NYUv2 intrinsics
                # if frame dimensions differ.
                K = Webot_K.copy()

                K[0, :] *= (
                    width / 640.0
                )

                K[1, :] *= (
                    height / 480.0
                )

                cv2.imwrite(
                    str(temp_image_path),
                    cv2.cvtColor(
                        rgb,
                        cv2.COLOR_RGB2BGR,
                    ),
                )

                inference = infer_da3(
                    model=model,
                    image_path=(
                        temp_image_path
                    ),
                    K_original=K,
                    original_width=width,
                    original_height=height,
                    process_res=(
                        args.process_res
                    ),
                )

                pred_depth = inference[
                    "depth_m"
                ]

                spatial_mask = (
                    make_spatial_mask(
                        height,
                        width,
                        args.crop,
                    )
                )

                range_masks = (
                    make_range_masks(
                        gt_depth,
                        spatial_mask,
                    )
                )


                if args.pc_cache_dir:
                    save_pointcloud_cache(
                        out_dir=args.pc_cache_dir,
                        dataset_index=dataset_index,
                        rgb=rgb,
                        gt_depth=gt_depth,
                        pred_depth=pred_depth,
                        K=K,
                        spatial_mask=spatial_mask,
                        stride=args.pc_stride,
                    )

                fps_values.append(
                    inference["fps"]
                )

                vram_values.append(
                    inference[
                        "peak_vram_gb"
                    ]
                )

                for (
                    range_name,
                    range_mask,
                ) in range_masks.items():

                    metrics = (
                        compute_metrics(
                            pred_depth,
                            gt_depth,
                            range_mask,
                        )
                    )

                    if metrics is None:
                        continue

                    rows.append({
                        "dataset_index": (
                            dataset_index
                        ),
                        "depth_range": (
                            range_name
                        ),
                        "valid_pixels": (
                            metrics[
                                "valid_pixels"
                            ]
                        ),
                        "mae_m": (
                            metrics[
                                "mae_m"
                            ]
                        ),
                        "mae_cm": (
                            metrics[
                                "mae_cm"
                            ]
                        ),
                        "absrel_pct": (
                            metrics[
                                "absrel_pct"
                            ]
                        ),
                        "delta1_pct": (
                            metrics[
                                "delta1_pct"
                            ]
                        ),
                        "fps": (
                            inference[
                                "fps"
                            ]
                        ),
                        "peak_vram_gb": (
                            inference[
                                "peak_vram_gb"
                            ]
                        ),
                        "input_resolution": (
                            f"{width}x{height}"
                        ),
                        "processed_resolution": (
                            f"{inference['processed_width']}"
                            f"x"
                            f"{inference['processed_height']}"
                        ),
                    })

                    pooled[
                        range_name
                    ].append({
                        "pred": (
                            metrics["_pred"]
                        ),
                        "gt": (
                            metrics["_gt"]
                        ),
                    })

                if (
                    position == 1
                    or position % 10 == 0
                    or position == len(indices)
                ):
                    print(
                        f"[{position:4d}/"
                        f"{len(indices):4d}] "
                        f"NYU index "
                        f"{dataset_index:4d} | "
                        f"FPS="
                        f"{inference['fps']:.2f} | "
                        f"VRAM="
                        f"{inference['peak_vram_gb']:.3f} GB"
                    )

        # ====================================================
        # Per-image CSV
        # ====================================================
        per_image_fields = [
            "dataset_index",
            "depth_range",
            "valid_pixels",
            "mae_m",
            "mae_cm",
            "absrel_pct",
            "delta1_pct",
            "fps",
            "peak_vram_gb",
            "input_resolution",
            "processed_resolution",
        ]

        with open(
            args.per_image_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=(
                    per_image_fields
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        # ====================================================
        # Exact pooled summary
        # ====================================================
        summary_rows = []

        for range_name in [
            "0-3m",
            "0-6m",
            "0-10m",
        ]:
            items = pooled[
                range_name
            ]

            if not items:
                continue

            pred = np.concatenate([
                item["pred"]
                for item in items
            ])

            gt = np.concatenate([
                item["gt"]
                for item in items
            ])

            abs_error = np.abs(
                pred - gt
            )

            ratio = np.maximum(
                pred / gt,
                gt / pred,
            )

            summary_rows.append({
                "depth_range": (
                    range_name
                ),
                "frames": (
                    len(indices)
                ),
                "valid_pixels": (
                    int(gt.size)
                ),
                "mae_m": float(
                    np.mean(abs_error)
                ),
                "mae_cm": float(
                    np.mean(abs_error)
                    * 100.0
                ),
                "absrel_pct": float(
                    np.mean(
                        abs_error / gt
                    )
                    * 100.0
                ),
                "delta1_pct": float(
                    np.mean(
                        ratio < 1.25
                    )
                    * 100.0
                ),
                "fps": float(
                    np.mean(
                        fps_values
                    )
                ),
                "peak_vram_gb": float(
                    np.max(
                        vram_values
                    )
                ),
            })

        summary_fields = [
            "depth_range",
            "frames",
            "valid_pixels",
            "mae_m",
            "mae_cm",
            "absrel_pct",
            "delta1_pct",
            "fps",
            "peak_vram_gb",
        ]

        with open(
            args.summary_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=(
                    summary_fields
                ),
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

        print()
        print("=" * 80)
        print(
            "FINAL POOLED RESULTS"
        )
        print("=" * 80)

        for row in summary_rows:
            print(
                f"{row['depth_range']:7s} | "
                f"MAE="
                f"{row['mae_cm']:7.2f} cm | "
                f"AbsRel="
                f"{row['absrel_pct']:6.2f}% | "
                f"delta1="
                f"{row['delta1_pct']:6.2f}% | "
                f"FPS="
                f"{row['fps']:.2f} | "
                f"VRAM="
                f"{row['peak_vram_gb']:.3f} GB"
            )

        print()
        print(
            f"Per-image CSV : "
            f"{Path(args.per_image_csv).resolve()}"
        )

        print(
            f"Summary CSV   : "
            f"{Path(args.summary_csv).resolve()}"
        )


        if args.pc_cache_dir:
            print(
                f"Point-cloud cache: "
                f"{Path(args.pc_cache_dir).resolve()}"
            )


if __name__ == "__main__":
    main()
