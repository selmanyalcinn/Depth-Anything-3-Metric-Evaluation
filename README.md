# Depth Anything 3 — Indoor & Outdoor Metric Depth Evaluation

A compact evaluation and visualization pipeline for **Depth Anything 3 (DA3METRIC-LARGE)** on indoor and outdoor metric-depth benchmarks.

The project evaluates the model on:

- **NYU Depth V2** — dense indoor RGB-D scenes
- **KITTI** — outdoor driving scenes with LiDAR-derived depth

In addition to numerical depth metrics, the project reconstructs corresponding ground-truth and predicted 3D point clouds so model errors can be inspected geometrically.

---

## Live Demo

A hosted Streamlit demo can be linked here after deployment:

**[Open the interactive demo]([STREAMLIT_DEMO_URL](https://depth-anything-3-metric-evaluation-27zb7vwudvfye73b4lpoif.streamlit.app))**

The hosted version can use a representative subset of cached NYUv2 and KITTI frames for fast interactive visualization, while the tables below report the full benchmark results.

---

## Model

This project uses **Depth Anything 3 — DA3METRIC-LARGE**, the metric-depth variant of Depth Anything 3.

Official model:

https://huggingface.co/depth-anything/DA3METRIC-LARGE

Official repository:

https://github.com/ByteDance-Seed/Depth-Anything-3

---

# Project Goal

The goal is to evaluate how accurately DA3METRIC-LARGE estimates real-world metric depth across two substantially different domains:

```text
NYUv2 → dense indoor / short-range depth
KITTI → sparse outdoor / long-range depth
```

For each dataset, the pipeline reports:

- **MAE** — Mean Absolute Error
- **AbsRel** — Absolute Relative Error
- **δ1** — percentage of valid predictions satisfying the 1.25× threshold
- **FPS**
- **Peak VRAM**

The project also exports same-pixel ground-truth and predicted 3D point clouds for visualization in a **Streamlit + Plotly** interface.

---

# Datasets

The full datasets and full point-cloud caches are not included in this repository.
A lightweight subset of cached NYUv2 and KITTI frames is included for running
the interactive demo without rerunning inference.

The full datasets should be downloaded separately from the official sources.

## NYU Depth V2

NYU Depth V2 is an indoor RGB-D dataset recorded using a Microsoft Kinect.

The labeled dataset contains:

- **1,449 aligned RGB-depth pairs**
- Dense metric depth maps
- 640 × 480 RGB-D data
- 464 indoor scenes

Official dataset page:

https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

Official labeled dataset download:

https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

The labeled dataset is distributed as a MATLAB `.mat` file and is approximately **2.8 GB**.

### NYUv2 evaluation ranges

```text
0–3 m
0–6 m
0–10 m
```

---

## KITTI Depth

KITTI is an outdoor autonomous-driving benchmark containing RGB images and LiDAR-derived metric depth.

This project uses the official manually selected validation data:

```text
val_selection_cropped
```

KITTI depth ground truth is much sparser than NYUv2 because it is derived from LiDAR measurements rather than a dense RGB-D sensor.

KITTI depth PNG values are decoded as:

```text
depth_m = pixel_value / 256.0
```

A value of `0` represents an invalid ground-truth pixel.

Official KITTI depth benchmark:

https://www.cvlibs.net/datasets/kitti/eval_depth_all.php

Official KITTI depth prediction benchmark:

https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

The relevant validation/test archive is:

```text
data_depth_selection.zip
```

KITTI may require registration or login before official benchmark data can be downloaded.

### KITTI evaluation ranges

```text
0–10 m
0–20 m
0–40 m
0–80 m
```

---

# Evaluation Protocol

Both datasets are evaluated using:

```text
Model       : DA3METRIC-LARGE
Process res : 1344
```

Metrics are computed only at pixels containing valid ground-truth depth.

No post-hoc scale alignment is applied.

For point-cloud visualization, ground-truth and predicted points are reconstructed from:

- the **same image pixel**
- the **same camera intrinsics**
- either the ground-truth depth or DA3-predicted depth as the Z coordinate

Therefore, every GT/prediction point pair has direct pixel-wise correspondence.

No ICP, geometric registration, or post-registration alignment is performed.

---

# Results

## NYU Depth V2 — Full Labeled Dataset

The complete **1,449-frame** labeled NYUv2 dataset was evaluated.

| Depth Range | Valid Pixels |          MAE |     AbsRel |         δ1 |
| ----------- | -----------: | -----------: | ---------: | ---------: |
| 0–3 m       |  288,918,478 | **26.22 cm** | **13.53%** | **89.87%** |
| 0–6 m       |  430,195,450 | **34.90 cm** | **13.39%** | **89.61%** |
| 0–10 m      |  445,132,800 | **38.57 cm** | **13.61%** | **89.05%** |

Performance on the evaluation machine:

```text
Average FPS : 0.288
Peak VRAM   : 3.348 GB
```

Main NYUv2 result:

```text
0–10 m
MAE    : 38.57 cm
AbsRel : 13.61%
δ1     : 89.05%
```

---

## KITTI — Full Outdoor Evaluation

The complete locally available `val_selection_cropped` split was evaluated using the same DA3 metric-depth pipeline.

Metrics were calculated only at pixels containing valid LiDAR ground truth.

### Final KITTI results

> Replace the table below with the values from `kitti_da3_metric_summary.csv`.

| Depth Range | Valid Pixels | MAE | AbsRel |  δ1 |
| ----------- | -----------: | --: | -----: | --: |
| 0–10 m      |            — |   — |      — |   — |
| 0–20 m      |            — |   — |      — |   — |
| 0–40 m      |            — |   — |      — |   — |
| 0–80 m      |            — |   — |      — |   — |

Full numerical results are stored in:

```text
kitti_da3_metric_summary.csv
kitti_da3_metric_per_image.csv
```

---

# Interactive Point-Cloud Viewer

The Streamlit application supports both:

```text
NYUv2 · Indoor
KITTI · Outdoor
```

For each cached frame, the viewer provides:

- Ground Truth vs Prediction
- Interactive 3D point clouds
- GT / Prediction overlay
- GT → Prediction displacement vectors
- Absolute depth-error visualization
- Relative depth-error visualization
- Per-frame numerical metrics
- Dataset-level summary metrics
- Best / worst frame sorting

Because the two point clouds are reconstructed from identical source pixels, their spatial separation represents direct metric-depth error rather than a registration mismatch.

---

# Running the Demo

If the repository contains the lightweight demo cache, the interactive viewer can be launched without downloading the full datasets or rerunning model inference.

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

## 2. Install UI dependencies

```bash
pip install numpy pandas streamlit plotly
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

## 3. Launch Streamlit

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

if Streamlit does not open the browser automatically.

---

# Reproducing the Full Evaluation

The full benchmark datasets are not included in the repository.

To reproduce the complete results:

1. Download NYU Depth V2 and/or KITTI from the official links above.
2. Install Depth Anything 3 and the evaluator dependencies.
3. Run the relevant evaluator.
4. Generate the full CSV metrics and `.npz` point-cloud caches.
5. Launch the Streamlit viewer.

## NYU Depth V2

```bash
python evaluate_nyu.py \
  --mat "datasets/nyu/nyu_depth_v2_labeled.mat" \
  --num_samples 0 \
  --process_res 1344 \
  --pc_cache_dir "nyu_da3_pc_cache"
```

## KITTI

```bash
python evaluate_kitti.py \
  --dataset "datasets/KITTI/val_selection_cropped" \
  --num_samples 0 \
  --process_res 1344 \
  --pc_stride 1 \
  --pc_cache_dir "kitti_da3_pc_cache" \
  --per_image_csv "kitti_da3_metric_per_image.csv" \
  --summary_csv "kitti_da3_metric_summary.csv"
```

Using:

```text
--num_samples 20
```

instead of:

```text
--num_samples 0
```

is recommended for a quick sanity test before processing the complete dataset.

---

# Conclusion

This project evaluates **DA3METRIC-LARGE** across both dense indoor RGB-D scenes and sparse outdoor LiDAR-based scenes using the same metric-depth methodology.

On the complete NYUv2 labeled dataset, DA3METRIC-LARGE achieved:

```text
MAE    : 38.57 cm
AbsRel : 13.61%
δ1     : 89.05%
```

over the full `0–10 m` evaluation range.

The KITTI evaluation extends the same analysis to outdoor driving scenes up to `80 m`.

Using the two datasets together gives a broader picture of the model's behavior across substantially different depth distributions and scene types.

The accompanying point-cloud viewer complements scalar benchmark metrics by showing how per-pixel depth errors translate into actual 3D geometric differences.
