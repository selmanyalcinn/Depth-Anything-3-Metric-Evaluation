from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="DA3 Indoor / Outdoor Point Cloud Inspector",
    layout="wide",
)

FRAME_RE = re.compile(r"frame_(\d+)\.npz$")


DATASETS = {
    "NYUv2 · Indoor": {
        "cache_dir": "nyu_da3_pc_cache",
        "metrics_csv": "nyu_da3_metric_per_image.csv",
        "primary_range": "0-10m",
        "range_label": "0–10 m",
        "mae_unit": "cm",
        "title": "NYUv2",
        "description": "Indoor RGB-D evaluation",
    },
    "KITTI · Outdoor": {
        "cache_dir": "kitti_da3_pc_cache",
        "metrics_csv": "kitti_da3_metric_per_image.csv",
        "primary_range": "0-80m",
        "range_label": "0–80 m",
        "mae_unit": "m",
        "title": "KITTI",
        "description": "Outdoor driving / LiDAR depth evaluation",
    },
}


def parse_index(path: Path):
    match = FRAME_RE.search(path.name)
    return int(match.group(1)) if match else None


@st.cache_data(show_spinner=False)
def load_cache(path_str: str):
    path = Path(path_str)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@st.cache_data(show_spinner=False)
def load_metrics(csv_path: str):
    path = Path(csv_path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def rgb_strings(colors):
    colors = np.asarray(colors, dtype=np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in colors]


def common_ranges(gt_points, pred_points):
    pts = np.vstack([gt_points, pred_points])

    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]

    if len(pts) == 0:
        return None

    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)

    pad = np.maximum(
        (hi - lo) * 0.03,
        1e-3,
    )

    return [
        (float(a - p), float(b + p))
        for a, b, p in zip(lo, hi, pad)
    ]


def base_layout(title, ranges=None):
    scene = dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z / depth (m)",
        aspectmode="data",
        camera=dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=0.0, y=0.0, z=-1.8),
        ),
    )

    if ranges is not None:
        scene["xaxis"] = dict(
            range=list(ranges[0]),
            title="X (m)",
        )
        scene["yaxis"] = dict(
            range=list(ranges[1]),
            title="Y (m)",
        )
        scene["zaxis"] = dict(
            range=list(ranges[2]),
            title="Z / depth (m)",
        )

    return dict(
        title=title,
        scene=scene,
        margin=dict(l=0, r=0, t=42, b=0),
        height=650,
        showlegend=True,
    )


def cloud_trace(
    points,
    colors,
    name,
    size=1.6,
    opacity=1.0,
):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color=rgb_strings(colors),
            opacity=opacity,
        ),
        hovertemplate=(
            "X=%{x:.3f} m<br>"
            "Y=%{y:.3f} m<br>"
            "Z=%{z:.3f} m"
            "<extra>" + name + "</extra>"
        ),
    )


def solid_cloud_trace(
    points,
    name,
    color,
    size=1.6,
    opacity=0.6,
):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        ),
        hovertemplate=(
            "X=%{x:.3f} m<br>"
            "Y=%{y:.3f} m<br>"
            "Z=%{z:.3f} m"
            "<extra>" + name + "</extra>"
        ),
    )


def error_cloud_trace(
    points,
    abs_error_m,
    rel_error,
    mode,
):
    if mode == "Absolute error":
        display = abs_error_m
        label = "Absolute error (m)"
        custom_abs = abs_error_m
    else:
        display = rel_error * 100.0
        label = "Relative error (%)"
        custom_abs = abs_error_m

    finite = display[np.isfinite(display)]

    vmax = (
        float(np.percentile(finite, 98))
        if len(finite)
        else 1.0
    )
    vmax = max(vmax, 1e-6)

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name="Prediction colored by error",
        marker=dict(
            size=1.8,
            color=display,
            colorscale="Turbo",
            cmin=0,
            cmax=vmax,
            colorbar=dict(title=label),
            opacity=1.0,
        ),
        customdata=np.column_stack([
            custom_abs,
            rel_error * 100.0,
        ]),
        hovertemplate=(
            "X=%{x:.3f} m<br>"
            "Y=%{y:.3f} m<br>"
            "Z=%{z:.3f} m"
            "<br>Abs error=%{customdata[0]:.3f} m"
            "<br>Rel error=%{customdata[1]:.2f}%"
            "<extra></extra>"
        ),
    )


def displacement_trace(
    gt_points,
    pred_points,
    max_vectors=300,
):
    n = len(gt_points)

    if n == 0:
        return None

    step = max(
        1,
        n // max_vectors,
    )

    indices = np.arange(
        0,
        n,
        step,
    )[:max_vectors]

    xs = []
    ys = []
    zs = []

    for i in indices:
        xs.extend([
            gt_points[i, 0],
            pred_points[i, 0],
            None,
        ])
        ys.extend([
            gt_points[i, 1],
            pred_points[i, 1],
            None,
        ])
        zs.extend([
            gt_points[i, 2],
            pred_points[i, 2],
            None,
        ])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="GT→Prediction displacement",
        line=dict(
            width=2,
            color="red",
        ),
        hoverinfo="skip",
    )


def subsample(data, max_points):
    n = len(data["gt_points"])

    if n <= max_points:
        return data

    indices = np.linspace(
        0,
        n - 1,
        max_points,
        dtype=np.int64,
    )

    output = {}

    for key, value in data.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and len(value) == n
        ):
            output[key] = value[indices]
        else:
            output[key] = value

    return output


def scalar_string(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    return str(value)


# ============================================================
# Sidebar: dataset and paths
# ============================================================
with st.sidebar:
    st.header("Dataset")

    dataset_name = st.selectbox(
        "Evaluation dataset",
        list(DATASETS.keys()),
    )

dataset_cfg = DATASETS[dataset_name]

with st.sidebar:
    st.caption(dataset_cfg["description"])


    cache_dir = st.text_input(
        "Point-cloud cache directory",
        dataset_cfg["cache_dir"],
        key=f"cache_{dataset_name}",
    )

    metrics_csv = st.text_input(
        "Per-image metrics CSV",
        dataset_cfg["metrics_csv"],
        key=f"csv_{dataset_name}",
    )

    max_points = st.select_slider(
        "Displayed points per cloud",
        options=[
            3000,
            5000,
            10000,
            15000,
            20000,
            30000,
            50000,
            75000,
        ],
        value=20000,
    )

    st.caption(
        "This only changes visualization load. "
        "It does not change cached or evaluated data."
    )


# ============================================================
# Locate cache
# ============================================================
cache_root = Path(cache_dir)

if not cache_root.is_dir():
    st.error(
        f"Cache directory not found: "
        f"{cache_root.resolve()}"
    )
    st.stop()

files = []

for path in cache_root.glob("frame_*.npz"):
    idx = parse_index(path)
    if idx is not None:
        files.append((idx, path))

files.sort(
    key=lambda item: item[0]
)

if not files:
    st.error(
        "No frame_XXXX.npz files found "
        "in the selected cache directory."
    )
    st.stop()


# ============================================================
# Metrics and ordering
# ============================================================
metrics = load_metrics(
    metrics_csv
)

primary_metrics = None

if (
    metrics is not None
    and {
        "dataset_index",
        "depth_range",
    }.issubset(metrics.columns)
):
    primary_metrics = metrics[
        metrics["depth_range"]
        == dataset_cfg["primary_range"]
    ].copy()

with st.sidebar:
    sort_mode = st.selectbox(
        "Frame order",
        [
            "Dataset index",
            "Worst AbsRel first",
            "Best AbsRel first",
            "Worst MAE first",
            "Worst δ1 first",
        ],
    )

file_map = {
    idx: path
    for idx, path in files
}

indices = [
    idx
    for idx, _ in files
]

if primary_metrics is not None:
    available = primary_metrics[
        primary_metrics[
            "dataset_index"
        ].isin(indices)
    ].copy()

    if sort_mode == "Worst AbsRel first":
        ordered = (
            available
            .sort_values(
                "absrel_pct",
                ascending=False,
            )["dataset_index"]
            .astype(int)
            .tolist()
        )

    elif sort_mode == "Best AbsRel first":
        ordered = (
            available
            .sort_values(
                "absrel_pct",
                ascending=True,
            )["dataset_index"]
            .astype(int)
            .tolist()
        )

    elif sort_mode == "Worst MAE first":
        mae_column = (
            "mae_m"
            if dataset_cfg["mae_unit"] == "m"
            else "mae_cm"
        )

        ordered = (
            available
            .sort_values(
                mae_column,
                ascending=False,
            )["dataset_index"]
            .astype(int)
            .tolist()
        )

    elif sort_mode == "Worst δ1 first":
        ordered = (
            available
            .sort_values(
                "delta1_pct",
                ascending=True,
            )["dataset_index"]
            .astype(int)
            .tolist()
        )

    else:
        ordered = indices

    # If metrics CSV is incomplete, keep cache-only frames too.
    missing = [
        idx
        for idx in indices
        if idx not in ordered
    ]
    ordered.extend(missing)

else:
    ordered = indices


with st.sidebar:
    frame_pos = st.slider(
        "Frame position",
        1,
        len(ordered),
        1,
    )

    selected_index = ordered[
        frame_pos - 1
    ]

    selected_index = st.selectbox(
        f"{dataset_cfg['title']} dataset index",
        ordered,
        index=frame_pos - 1,
        key=f"frame_select_{dataset_name}",
    )


# ============================================================
# Load selected frame
# ============================================================
raw = load_cache(
    str(file_map[selected_index])
)

data = subsample(
    raw,
    max_points,
)

gt = np.asarray(
    data["gt_points"],
    dtype=np.float32,
)

pred = np.asarray(
    data["pred_points"],
    dtype=np.float32,
)

colors = np.asarray(
    data["colors"],
    dtype=np.uint8,
)

abs_error = np.asarray(
    data["abs_error_m"],
    dtype=np.float32,
)

rel_error = np.asarray(
    data["rel_error"],
    dtype=np.float32,
)

ranges = common_ranges(
    gt,
    pred,
)


# ============================================================
# Header
# ============================================================
st.title(
    "Depth Anything 3 · Point Cloud Inspector"
)

st.caption(
    "Ground-truth and DA3 point clouds use the same source pixels "
    "and the same camera intrinsics. "
    "No ICP, scale alignment, or post-registration is applied."
)

header_cols = st.columns([2, 2, 2])

header_cols[0].metric(
    "Dataset",
    dataset_cfg["title"],
)

header_cols[1].metric(
    "Cached frames",
    f"{len(files):,}",
)

header_cols[2].metric(
    "Selected dataset index",
    selected_index,
)

image_name = None

if "image_name" in raw:
    image_name = scalar_string(
        raw["image_name"]
    )


# ============================================================
# Metrics
# ============================================================
if metrics is not None:
    frame_metrics = metrics[
        metrics["dataset_index"]
        == selected_index
    ].copy()

    if not frame_metrics.empty:
        primary = frame_metrics[
            frame_metrics["depth_range"]
            == dataset_cfg["primary_range"]
        ]

        if not primary.empty:
            row = primary.iloc[0]

            cols = st.columns(4)

            if dataset_cfg["mae_unit"] == "m":
                mae_text = (
                    f"{row['mae_m']:.3f} m"
                )
            else:
                mae_text = (
                    f"{row['mae_cm']:.2f} cm"
                )

            cols[0].metric(
                f"MAE · {dataset_cfg['range_label']}",
                mae_text,
            )

            cols[1].metric(
                f"AbsRel · {dataset_cfg['range_label']}",
                f"{row['absrel_pct']:.2f}%",
            )

            cols[2].metric(
                f"δ1 · {dataset_cfg['range_label']}",
                f"{row['delta1_pct']:.2f}%",
            )

            if "fps" in row:
                cols[3].metric(
                    "Inference FPS",
                    f"{row['fps']:.2f}",
                )

        with st.expander(
            "All numerical metrics for this frame"
        ):
            show_cols = [
                column
                for column in [
                    "depth_range",
                    "valid_pixels",
                    "mae_m",
                    "mae_cm",
                    "absrel_pct",
                    "delta1_pct",
                    "fps",
                    "peak_vram_gb",
                ]
                if column in frame_metrics.columns
            ]

            st.dataframe(
                frame_metrics[show_cols],
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# Correspondence statistics
# ============================================================
pointwise_3d = np.linalg.norm(
    pred - gt,
    axis=1,
)

stat_cols = st.columns(4)

stat_cols[0].metric(
    "Displayed correspondences",
    f"{len(gt):,}",
)

stat_cols[1].metric(
    "Mean 3D displacement",
    f"{np.mean(pointwise_3d):.3f} m",
)

stat_cols[2].metric(
    "Median depth error",
    f"{np.median(abs_error):.3f} m",
)

stat_cols[3].metric(
    "95th pct depth error",
    f"{np.percentile(abs_error, 95):.3f} m",
)


# ============================================================
# Point cloud tabs
# ============================================================
side_tab, overlay_tab, error_tab = st.tabs([
    "GT vs Prediction",
    "Overlay",
    "Error cloud",
])


with side_tab:
    c1, c2 = st.columns(2)

    fig_gt = go.Figure(
        data=[
            cloud_trace(
                gt,
                colors,
                "Ground truth",
            )
        ]
    )

    fig_gt.update_layout(
        **base_layout(
            "Ground-truth point cloud",
            ranges,
        )
    )

    c1.plotly_chart(
        fig_gt,
        use_container_width=True,
        config={
            "scrollZoom": True,
        },
    )

    fig_pred = go.Figure(
        data=[
            cloud_trace(
                pred,
                colors,
                "DA3 prediction",
            )
        ]
    )

    fig_pred.update_layout(
        **base_layout(
            "DA3 predicted point cloud",
            ranges,
        )
    )

    c2.plotly_chart(
        fig_pred,
        use_container_width=True,
        config={
            "scrollZoom": True,
        },
    )

with overlay_tab:
    show_vectors = st.checkbox(
        "Show GT→prediction displacement vectors",
        value=False,
    )

    fig = go.Figure()

    fig.add_trace(
        solid_cloud_trace(
            gt,
            "Ground truth",
            "#2E86DE",
            opacity=0.50,
        )
    )

    fig.add_trace(
        solid_cloud_trace(
            pred,
            "DA3 prediction",
            "#F39C12",
            opacity=0.50,
        )
    )

    if show_vectors:
        vector_trace = displacement_trace(
            gt,
            pred,
        )

        if vector_trace is not None:
            fig.add_trace(
                vector_trace
            )

    fig.update_layout(
        **base_layout(
            "GT / DA3 overlay",
            ranges,
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
        },
    )

    st.caption(
        "Each GT point and prediction point comes from the same image pixel. "
        "Their separation is therefore a direct correspondence error."
    )


with error_tab:
    error_mode = st.radio(
        "Color by",
        [
            "Absolute error",
            "Relative error (%)",
        ],
        horizontal=True,
    )

    high_only = st.checkbox(
        "Show only high-error points",
        value=False,
    )

    mask = np.ones(
        len(pred),
        dtype=bool,
    )

    if high_only:
        if error_mode == "Absolute error":
            errors = abs_error

            default_threshold = float(
                np.percentile(
                    errors,
                    75,
                )
            )

            upper = float(
                max(
                    np.percentile(
                        errors,
                        99,
                    ),
                    0.01,
                )
            )

            threshold = st.slider(
                "Minimum absolute error (m)",
                0.0,
                upper,
                min(
                    default_threshold,
                    upper,
                ),
            )

            mask = (
                errors >= threshold
            )

        else:
            errors = rel_error * 100.0

            default_threshold = float(
                np.percentile(
                    errors,
                    75,
                )
            )

            upper = float(
                max(
                    np.percentile(
                        errors,
                        99,
                    ),
                    1.0,
                )
            )

            threshold = st.slider(
                "Minimum relative error (%)",
                0.0,
                upper,
                min(
                    default_threshold,
                    upper,
                ),
            )

            mask = (
                errors >= threshold
            )

    fig_error = go.Figure(
        data=[
            error_cloud_trace(
                pred[mask],
                abs_error[mask],
                rel_error[mask],
                error_mode,
            )
        ]
    )

    fig_error.update_layout(
        **base_layout(
            "DA3 prediction colored by per-pixel depth error",
            ranges,
        )
    )

    st.plotly_chart(
        fig_error,
        use_container_width=True,
        config={
            "scrollZoom": True,
        },
    )


st.divider()

st.caption(
    f"{dataset_cfg['title']} · "
    f"{len(files)} cached frames · "
    f"Primary evaluation range: "
    f"{dataset_cfg['range_label']}"
)
