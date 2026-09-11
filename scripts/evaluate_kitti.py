import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def read_kitti_depth_png(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read GT depth: {path}")
    if depth.dtype != np.uint16:
        print(f"[WARN] {path.name}: expected uint16 GT, got {depth.dtype}")
    return depth.astype(np.float32) / 256.0


def read_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(str(path), dtype=np.float32)
    if values.shape == (3, 3):
        return values
    values = values.reshape(-1)
    if values.size != 9:
        raise ValueError(f"Unexpected intrinsics format in {path}: {values.shape}")
    return values.reshape(3, 3)


def find_samples(root: Path):
    image_dir = root / "image"
    gt_dir = root / "groundtruth_depth"
    intr_dir = root / "intrinsics"

    for d in (image_dir, gt_dir, intr_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Required directory not found: {d.resolve()}")

    samples = []
    for dataset_index, image_path in enumerate(sorted(image_dir.glob("*.png"))):
        gt_name = image_path.name.replace(
            "_sync_image_",
            "_sync_groundtruth_depth_",
        )
        gt_path = gt_dir / gt_name
        intr_path = intr_dir / f"{image_path.stem}.txt"

        if gt_path.exists() and intr_path.exists():
            samples.append({
                "dataset_index": dataset_index,
                "image_path": image_path,
                "gt_path": gt_path,
                "intr_path": intr_path,
            })

    return samples


def deterministic_subset(samples, n):
    if n <= 0 or n >= len(samples):
        return samples
    indices = np.linspace(0, len(samples) - 1, n, dtype=int)
    return [samples[i] for i in indices]


def range_masks(gt_depth_m):
    base = np.isfinite(gt_depth_m) & (gt_depth_m >= 0.1)
    return {
        "0-10m": base & (gt_depth_m <= 10.0),
        "0-20m": base & (gt_depth_m <= 20.0),
        "0-40m": base & (gt_depth_m <= 40.0),
        "0-80m": base & (gt_depth_m <= 80.0),
    }


def compute_metric_sums(pred_m, gt_m, valid):
    valid = (
        valid
        & np.isfinite(pred_m)
        & (pred_m > 0)
        & np.isfinite(gt_m)
        & (gt_m > 0)
    )

    count = int(valid.sum())
    if count == 0:
        return None

    pred = pred_m[valid].astype(np.float64)
    gt = gt_m[valid].astype(np.float64)

    abs_err = np.abs(pred - gt)
    rel_err = abs_err / gt
    ratio = np.maximum(pred / gt, gt / pred)

    return {
        "valid_pixels": count,
        "sum_abs_error": float(np.sum(abs_err)),
        "sum_rel_error": float(np.sum(rel_err)),
        "delta1_hits": int(np.sum(ratio < 1.25)),
        "mae_m": float(np.mean(abs_err)),
        "mae_cm": float(np.mean(abs_err) * 100.0),
        "absrel_pct": float(np.mean(rel_err) * 100.0),
        "delta1_pct": float(np.mean(ratio < 1.25) * 100.0),
    }


def empty_pool():
    return {
        "valid_pixels": 0,
        "sum_abs_error": 0.0,
        "sum_rel_error": 0.0,
        "delta1_hits": 0,
    }


def update_pool(pool, metrics):
    pool["valid_pixels"] += metrics["valid_pixels"]
    pool["sum_abs_error"] += metrics["sum_abs_error"]
    pool["sum_rel_error"] += metrics["sum_rel_error"]
    pool["delta1_hits"] += metrics["delta1_hits"]


def build_corresponding_point_clouds(
    rgb,
    gt_depth,
    pred_depth,
    K,
    max_depth=80.0,
    stride=1,
):
    valid = (
        np.isfinite(gt_depth)
        & (gt_depth >= 0.1)
        & (gt_depth <= max_depth)
        & np.isfinite(pred_depth)
        & (pred_depth > 0)
    )

    if stride > 1:
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

    x_gt = (u.astype(np.float32) - cx) * z_gt / fx
    y_gt = -(v.astype(np.float32) - cy) * z_gt / fy

    x_pred = (u.astype(np.float32) - cx) * z_pred / fx
    y_pred = -(v.astype(np.float32) - cy) * z_pred / fy

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
    image_name,
    rgb,
    gt_depth,
    pred_depth,
    K,
    stride,
):
    pc = build_corresponding_point_clouds(
        rgb=rgb,
        gt_depth=gt_depth,
        pred_depth=pred_depth,
        K=K,
        max_depth=80.0,
        stride=stride,
    )

    if pc is None:
        print(f"[WARN] No point-cloud pixels for {image_name}")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / f"frame_{dataset_index:04d}.npz",
        dataset=np.array("KITTI"),
        dataset_index=np.int32(dataset_index),
        image_name=np.array(image_name),
        K=K.astype(np.float32),
        **pc,
    )


def infer_da3(
    model,
    image_path,
    K_original,
    original_width,
    original_height,
    process_res,
):
    K_batch = K_original[None, ...].astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.inference_mode():
        prediction = model.inference(
            image=[str(image_path)],
            intrinsics=K_batch,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    raw_depth = np.asarray(prediction.depth[0], dtype=np.float32)
    processed_h, processed_w = raw_depth.shape

    pred_intrinsics = getattr(prediction, "intrinsics", None)

    if pred_intrinsics is not None:
        K_processed = np.asarray(pred_intrinsics[0], dtype=np.float32)
        fx_processed = float(K_processed[0, 0])
        fy_processed = float(K_processed[1, 1])
    else:
        fx_processed = float(K_original[0, 0]) * processed_w / original_width
        fy_processed = float(K_original[1, 1]) * processed_h / original_height

    focal_processed = 0.5 * (fx_processed + fy_processed)

    pred_depth_processed_m = (
        focal_processed * raw_depth / 300.0
    ).astype(np.float32)

    pred_depth_m = cv2.resize(
        pred_depth_processed_m,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    return {
        "depth_m": pred_depth_m,
        "fps": fps,
        "peak_vram_gb": peak_vram_gb,
        "processed_width": processed_w,
        "processed_height": processed_h,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DA3METRIC-LARGE on KITTI and export "
            "same-pixel GT/pred point-cloud caches."
        )
    )

    parser.add_argument("--dataset", type=str, default="KITTI/val_selection_cropped")
    parser.add_argument("--model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--num_samples", type=int, default=20, help="0 = all matched KITTI samples.")
    parser.add_argument("--process_res", type=int, default=1344)
    parser.add_argument("--per_image_csv", type=str, default="kitti_da3_metric_per_image.csv")
    parser.add_argument("--summary_csv", type=str, default="kitti_da3_metric_summary.csv")
    parser.add_argument("--pc_cache_dir", type=str, default="kitti_da3_pc_cache")
    parser.add_argument(
        "--pc_stride",
        type=int,
        default=1,
        help="KITTI GT is sparse; 1 is recommended for initial tests.",
    )

    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root.resolve()}")

    all_samples = find_samples(root)
    if not all_samples:
        raise RuntimeError(
            f"No matched image/GT/intrinsics samples found under {root.resolve()}"
        )

    samples = deterministic_subset(all_samples, args.num_samples)

    print("=" * 88)
    print("DEPTH ANYTHING 3 - KITTI METRIC EVALUATION + POINT CLOUD CACHE")
    print("=" * 88)
    print(f"Dataset            : {root.resolve()}")
    print(f"Available pairs    : {len(all_samples)}")
    print(f"Evaluating samples : {len(samples)}")
    print(f"Process resolution : {args.process_res}")
    print(f"Point-cloud stride : {args.pc_stride}")
    print(f"Cache directory    : {Path(args.pc_cache_dir).resolve()}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()
    print("Model loaded.\n")

    warm = samples[0]
    K_warm = read_intrinsics(warm["intr_path"])[None, ...]

    print("Warm-up inference...")
    with torch.inference_mode():
        _ = model.inference(
            image=[str(warm["image_path"])],
            intrinsics=K_warm,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("Warm-up done.\n")

    rows = []

    pooled = {
        "0-10m": empty_pool(),
        "0-20m": empty_pool(),
        "0-40m": empty_pool(),
        "0-80m": empty_pool(),
    }

    fps_values = []
    vram_values = []

    for position, sample in enumerate(samples, 1):
        dataset_index = int(sample["dataset_index"])
        image_path = sample["image_path"]
        gt_path = sample["gt_path"]
        intr_path = sample["intr_path"]

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb.shape[:2]

        gt_depth_m = read_kitti_depth_png(gt_path)

        if gt_depth_m.shape != (original_h, original_w):
            raise RuntimeError(
                f"Image/GT size mismatch for {image_path.name}: "
                f"image={(original_h, original_w)}, GT={gt_depth_m.shape}"
            )

        K_original = read_intrinsics(intr_path)

        inference = infer_da3(
            model=model,
            image_path=image_path,
            K_original=K_original,
            original_width=original_w,
            original_height=original_h,
            process_res=args.process_res,
        )

        pred_depth_m = inference["depth_m"]

        save_pointcloud_cache(
            out_dir=args.pc_cache_dir,
            dataset_index=dataset_index,
            image_name=image_path.name,
            rgb=rgb,
            gt_depth=gt_depth_m,
            pred_depth=pred_depth_m,
            K=K_original,
            stride=args.pc_stride,
        )

        fps_values.append(inference["fps"])
        vram_values.append(inference["peak_vram_gb"])

        masks = range_masks(gt_depth_m)

        for range_name, valid in masks.items():
            metrics = compute_metric_sums(
                pred_depth_m,
                gt_depth_m,
                valid,
            )

            if metrics is None:
                continue

            update_pool(pooled[range_name], metrics)

            rows.append({
                "dataset_index": dataset_index,
                "image": image_path.name,
                "depth_range": range_name,
                "valid_pixels": metrics["valid_pixels"],
                "mae_m": metrics["mae_m"],
                "mae_cm": metrics["mae_cm"],
                "absrel_pct": metrics["absrel_pct"],
                "delta1_pct": metrics["delta1_pct"],
                "fps": inference["fps"],
                "peak_vram_gb": inference["peak_vram_gb"],
                "input_resolution": f"{original_w}x{original_h}",
                "processed_resolution": (
                    f"{inference['processed_width']}x"
                    f"{inference['processed_height']}"
                ),
            })

        print(
            f"[{position:4d}/{len(samples):4d}] "
            f"KITTI index {dataset_index:4d} | "
            f"{image_path.name} | "
            f"FPS={inference['fps']:.2f} | "
            f"VRAM={inference['peak_vram_gb']:.3f} GB"
        )

    per_image_fields = [
        "dataset_index",
        "image",
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

    with open(args.per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for range_name in ["0-10m", "0-20m", "0-40m", "0-80m"]:
        p = pooled[range_name]
        n = p["valid_pixels"]
        if n == 0:
            continue

        summary_rows.append({
            "depth_range": range_name,
            "frames": len(samples),
            "valid_pixels": n,
            "mae_m": p["sum_abs_error"] / n,
            "mae_cm": (p["sum_abs_error"] / n) * 100.0,
            "absrel_pct": (p["sum_rel_error"] / n) * 100.0,
            "delta1_pct": (p["delta1_hits"] / n) * 100.0,
            "fps": float(np.mean(fps_values)),
            "peak_vram_gb": float(np.max(vram_values)),
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

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 88)
    print("FINAL POOLED RESULTS")
    print("=" * 88)

    for row in summary_rows:
        print(
            f"{row['depth_range']:7s} | "
            f"MAE={row['mae_m']:7.3f} m | "
            f"AbsRel={row['absrel_pct']:6.2f}% | "
            f"delta1={row['delta1_pct']:6.2f}% | "
            f"FPS={row['fps']:.2f} | "
            f"VRAM={row['peak_vram_gb']:.3f} GB"
        )

    print()
    print(f"Per-image CSV : {Path(args.per_image_csv).resolve()}")
    print(f"Summary CSV   : {Path(args.summary_csv).resolve()}")
    print(f"Point-cloud cache: {Path(args.pc_cache_dir).resolve()}")


if __name__ == "__main__":
    main()
