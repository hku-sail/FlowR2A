#!/usr/bin/env python3
"""Interactive Gradio inspector for the navtrain reward cache.

Visualizes the pre-computed reward cache (`transfuser_reward.gz`, rewards for all
8192 vocab trajectories per token) on BEV with reward-based coloring. Also previews
the training-time trajectory sampling and flow-matching noising. No agent loading
or inference needed.

Usage:
    python scripts/visualization/interactive_train_inspector.py --port 7860

    # With explicit paths:
    python scripts/visualization/interactive_train_inspector.py \
        --cache-path $NAVSIM_EXP_ROOT/training_cache \
        --vocab-path ckpts/8192.npy \
        --clip-index /path/to/navtrain_clips_index.pkl \
        --port 7860 --share

Without `--clip-index`, tokens are grouped into clips by their scene directory.
"""

import argparse
import gzip
import hashlib
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.config import NEW_TAB_10
from navsim.visualization.plots import plot_bev_frame

from inspector_common import (
    clip_info, draw_gt, draw_scored_trajs, draw_time_markers, fig_to_pil,
    load_clip_index, render_cached_features, resolve_token, save_result, sort_clips,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ROOT = os.environ["NAVSIM_EXP_ROOT"]
DEFAULT_CACHE_PATH = os.path.join(EXP_ROOT, "training_cache")
DEFAULT_VOCAB_PATH = "ckpts/8192.npy"
SAVE_DIR = "viz_output/train_inspector"

OPENSCENE_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
NAVSIM_LOG_PATH = OPENSCENE_ROOT / "navsim_logs" / "trainval"
SENSOR_BLOBS_PATH = OPENSCENE_ROOT / "sensor_blobs" / "trainval"

# 40-step vocab (0.1s) -> 8 poses (0.5s), same as traj_list[:, 4::5] in the model
DOWNSAMPLE_INDICES = np.array([4, 9, 14, 19, 24, 29, 34, 39])

# Reward-cache keys offered for visualization, in dropdown order. The first group
# are per-trajectory scalars (N,); the last three are per-step arrays (N, 40[, 2]).
SCALAR_REWARD_KEYS = [
    "pdm_score", "no_at_fault_collisions", "drivable_area_compliance",
    "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
    "time_to_collision_within_bound", "lane_keeping", "history_comfort",
    "prob_density", "safe_ego_progress_normed",
]
ARRAY_REWARD_KEYS = ["ego_areas", "ttc_time", "ttc_time_b15"]

# Trajectory sampling, aligned with TrajectoryHead.traj_sampling in
# navsim/agents/flowr2a/flowr2a_model.py (inverse prob_density weighting)
SAMPLING_ALPHA = 0.6

# Trajectory normalization, aligned with TrajectoryHead.norm_odo
NORM_RANGE = 2
NORM_X_OFFSET, NORM_X_SPAN = 1.2, 56.9
NORM_Y_OFFSET, NORM_Y_SPAN = 20, 46


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
vocab_8: Optional[np.ndarray] = None       # (N, 8, 3) downsampled
token_to_cache_path: Dict[str, Path] = {}
clips: List[Dict] = []
_current: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reward_cache(token_path: Path) -> Dict[str, Any]:
    with gzip.open(token_path / "transfuser_reward.gz", "rb") as f:
        return pickle.load(f)


def build_token_index(cache_path: str) -> Dict[str, Path]:
    """Scan cache directory for tokens that have reward caches.

    Caches the result to viz_output/token_index_{hash}.pkl so subsequent
    launches skip the slow directory walk.
    """
    cache_root = Path(cache_path)
    path_hash = hashlib.md5(str(cache_root.resolve()).encode()).hexdigest()[:10]
    index_cache = Path("viz_output") / f"token_index_{path_hash}.pkl"

    if index_cache.exists():
        with open(index_cache, "rb") as f:
            result = {tok: Path(p) for tok, p in pickle.load(f).items()}
        print(f"Loaded {len(result)} tokens from cached index {index_cache}")
        return result

    result = {}
    for scene_dir in tqdm(sorted(cache_root.iterdir()), desc="Scanning cache"):
        if not scene_dir.is_dir():
            continue
        for token_dir in scene_dir.iterdir():
            if (token_dir / "transfuser_reward.gz").exists():
                result[token_dir.name] = token_dir
    print(f"Found {len(result)} tokens with reward caches")

    # Stored as {token: str(path)} to stay portable
    index_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(index_cache, "wb") as f:
        pickle.dump({tok: str(p) for tok, p in result.items()}, f)
    print(f"Saved token index to {index_cache}")
    return result


def build_fallback_clips(token_to_path: Dict[str, Path]) -> List[Dict]:
    """Group tokens by scene directory as a fallback clip index."""
    scene_to_tokens = defaultdict(list)
    for token, path in token_to_path.items():
        scene_to_tokens[path.parent.name].append(token)
    return sort_clips([{"log_name": name, "tokens": sorted(tokens)}
                       for name, tokens in sorted(scene_to_tokens.items())])


def load_scene_for_token(token: str) -> Any:
    log_name = token_to_cache_path[token].parent.name
    sf = SceneFilter(
        tokens=[token], log_names=[log_name],
        num_history_frames=2, num_future_frames=19, frame_interval=1,
    )
    sl = SceneLoader(
        data_path=NAVSIM_LOG_PATH, sensor_blobs_path=SENSOR_BLOBS_PATH,
        scene_filter=sf, sensor_config=SensorConfig.build_no_sensors(),
    )
    return sl.get_scene_from_token(token)


# ---------------------------------------------------------------------------
# Trajectory sampling & flow matching noise (standalone, no model dependency)
# ---------------------------------------------------------------------------

def norm_odo(traj: np.ndarray) -> np.ndarray:
    """Normalize trajectories: (*, 3) [x,y,heading] -> (*, 4) [x,y,sin,cos]."""
    x = 2 * NORM_RANGE * (traj[..., 0:1] + NORM_X_OFFSET) / NORM_X_SPAN - NORM_RANGE
    y = 2 * NORM_RANGE * (traj[..., 1:2] + NORM_Y_OFFSET) / NORM_Y_SPAN - NORM_RANGE
    return np.concatenate([x, y, np.sin(traj[..., 2:3]), np.cos(traj[..., 2:3])], axis=-1).astype(np.float32)


def denorm_odo(normed: np.ndarray) -> np.ndarray:
    """Denormalize: (*, 4) [x,y,sin,cos] -> (*, 3) [x,y,heading]."""
    x = (normed[..., 0:1] + NORM_RANGE) / (2 * NORM_RANGE) * NORM_X_SPAN - NORM_X_OFFSET
    y = (normed[..., 1:2] + NORM_RANGE) / (2 * NORM_RANGE) * NORM_Y_SPAN - NORM_Y_OFFSET
    heading = np.arctan2(normed[..., 2:3], normed[..., 3:4])
    return np.concatenate([x, y, heading], axis=-1).astype(np.float32)


def sample_trajs(reward: Dict[str, Any], num: int, alpha: float) -> np.ndarray:
    """Sample vocab indices with weights (1 / prob_density) ** alpha, as in TrajectoryHead.traj_sampling."""
    weights = np.power(1.0 / (np.asarray(reward["prob_density"], dtype=np.float64) + 1e-8), alpha)
    return np.random.choice(len(vocab_8), size=num, replace=True, p=weights / weights.sum())


def add_flow_noise(trajs: np.ndarray, t: float, seed: Optional[int] = None) -> np.ndarray:
    """Flow-matching noising x_t = t * x_0 + (1 - t) * noise in normalized space.

    t=1 is clean, t=0 is pure noise. Returns (K, 8, 3) noised trajectories.
    """
    rng = np.random.default_rng(seed)
    x_0 = norm_odo(trajs)
    noise = rng.standard_normal(x_0.shape).astype(np.float32)
    return denorm_odo(t * x_0 + (1 - t) * noise)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def bev_canvas(scene, show_gt=True, traj_only=False):
    """BEV figure with the map, or a plain canvas fitted to whatever is drawn."""
    if traj_only:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_aspect("equal")
        ax.invert_xaxis()  # BEV convention: +y (left) is drawn to the left
        ax.margins(0.05)
        ax.axis("off")
    else:
        fig, ax = plot_bev_frame(scene, scene.scene_metadata.num_history_frames - 1)
    if show_gt:
        draw_gt(ax, scene)
    return fig, ax


def ego_area_color(non_drivable: bool, oncoming: bool) -> str:
    if non_drivable and oncoming:
        return "#9467bd"       # Purple - both
    if oncoming:
        return NEW_TAB_10[2]   # Red - oncoming
    if non_drivable:
        return NEW_TAB_10[1]   # Orange - non-drivable
    return NEW_TAB_10[4]       # Green - safe


def ttc_color(ttc: float) -> str:
    if ttc >= 1.5:
        return NEW_TAB_10[4]   # Green - safe
    if ttc >= 0.9:
        return NEW_TAB_10[5]   # Yellow - warning
    if ttc >= 0.3:
        return NEW_TAB_10[1]   # Orange - caution
    return NEW_TAB_10[2]       # Red - collision


def draw_segments(ax, traj, colors, lw, show_markers):
    """Draw an (8, 3) trajectory segment by segment, segment s in colors[s]."""
    poses = np.vstack([[0, 0, 0], traj])
    for s, color in enumerate(colors):
        ax.plot(poses[s:s + 2, 1], poses[s:s + 2, 0], color=color,
                linewidth=lw, zorder=10, alpha=0.9)
    if show_markers:
        draw_time_markers(ax, traj)


def add_legend(ax, items: List[Tuple[str, str]]):
    leg = ax.legend(handles=[Patch(facecolor=c, label=label) for c, label in items],
                    loc="upper right", fontsize=8)
    leg.set_zorder(100)


def render_scalar(ax, values, top_k, color_mode, lw, show_markers) -> str:
    """Trajectories colored by a per-trajectory scalar reward, best on top."""
    scores = np.asarray(values, dtype=np.float64)
    shown = draw_scored_trajs(ax, vocab_8, scores, top_k=top_k, color_mode=color_mode,
                              lw=lw, show_markers=show_markers)
    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        return "No valid trajectories to display."
    return "\n".join([
        f"Showing {len(shown)} / {len(valid)} valid trajectories (total: {len(scores)})",
        f"Score range (all valid): [{valid.min():.4f}, {valid.max():.4f}]",
        f"Score mean / median (all valid): {valid.mean():.4f} / {np.median(valid):.4f}",
        f"Score range (shown): [{scores[shown].min():.4f}, {scores[shown].max():.4f}]",
    ])


def render_ego_areas(ax, ego_areas_full, top_k, prefer_safe, lw, show_markers) -> str:
    """Per-segment non-drivable / oncoming-lane coloring.

    top_k trajectories are drawn (0 = all), picking the most-violating ones
    first, or the safest ones when prefer_safe.
    """
    ego_areas = np.asarray(ego_areas_full)
    if ego_areas.shape[1] == 40:
        ego_areas = ego_areas[:, DOWNSAMPLE_INDICES]   # (N, 8, 2)
    violations = ego_areas.sum(axis=(1, 2))
    order = np.argsort(violations) if prefer_safe else np.argsort(-violations)
    selected = order[:top_k] if top_k > 0 else order

    # Draw safest first so violation segments end up on top
    for i in sorted(selected, key=lambda i: violations[i]):
        colors = [ego_area_color(bool(nd), bool(on)) for nd, on in ego_areas[i]]
        draw_segments(ax, vocab_8[i], colors, lw, show_markers)
    add_legend(ax, [(NEW_TAB_10[4], "Safe"), (NEW_TAB_10[1], "Non-drivable"),
                    (NEW_TAB_10[2], "Oncoming"), ("#9467bd", "Both")])

    return "\n".join([
        f"Showing {len(selected)} trajectories (total: {len(ego_areas)})",
        f"With violations: {int((violations > 0).sum())} / {len(ego_areas)}",
        f"  Non-drivable: {int(ego_areas[:, :, 0].any(axis=1).sum())}",
        f"  Oncoming: {int(ego_areas[:, :, 1].any(axis=1).sum())}",
    ])


def render_ttc(ax, ttc_full, top_k, prefer_safe, lw, show_markers) -> str:
    """Per-segment time-to-collision coloring (segment = min TTC over its 5 steps).

    top_k trajectories are drawn (0 = all), picking the earliest-collision ones
    first, or the safest ones when prefer_safe.
    """
    ttc = np.asarray(ttc_full, dtype=np.float32)       # (N, 40)
    min_ttc = ttc.min(axis=1)
    if prefer_safe:
        order = np.argsort(-min_ttc)
    else:
        below = ttc < 1.5
        first_collision = np.where(below.any(axis=1), below.argmax(axis=1), ttc.shape[1])
        order = np.argsort(first_collision)
    selected = order[:top_k] if top_k > 0 else order

    seg_start = np.concatenate([[0], DOWNSAMPLE_INDICES[:-1] + 1])
    seg_ttc = np.stack([ttc[:, s:e + 1].min(axis=1) for s, e in zip(seg_start, DOWNSAMPLE_INDICES)], axis=1)

    # Draw safest first so collision segments end up on top
    for i in sorted(selected, key=lambda i: -min_ttc[i]):
        draw_segments(ax, vocab_8[i], [ttc_color(float(v)) for v in seg_ttc[i]], lw, show_markers)
    add_legend(ax, [(NEW_TAB_10[4], "TTC >= 1.5s (safe)"), (NEW_TAB_10[5], "TTC >= 0.9s"),
                    (NEW_TAB_10[1], "TTC >= 0.3s"), (NEW_TAB_10[2], "TTC < 0.3s")])

    return "\n".join([
        f"Showing {len(selected)} trajectories (total: {len(ttc)})",
        f"With collision (min TTC < 1.5s): {int((min_ttc < 1.5).sum())} / {len(ttc)}",
        f"Min TTC overall: {min_ttc.min():.3f}s",
        f"Mean min-TTC: {min_ttc.mean():.3f}s",
    ])


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def _finish(fig, dpi, compact, info: str, settings: Dict[str, Any]):
    """Render fig and remember it (with its settings) for saving."""
    bev = fig_to_pil(fig, dpi=int(dpi), compact=compact)
    _current.update(last_bev=bev, last_info=info,
                    last_settings={"log_name": _current["log_name"], **settings})
    return bev, info


def load_scene_callback(token_text, clip_idx, frame_idx, dpi, show_gt, compact):
    token = resolve_token(token_text, clip_idx, frame_idx, clips, token_to_cache_path)
    if token is None:
        return None, "Token not found.", gr.update()
    try:
        scene = load_scene_for_token(token)
        reward = load_reward_cache(token_to_cache_path[token])
    except Exception as e:
        return None, f"Error loading scene: {e}", gr.update()

    _current.clear()
    _current.update(token=token, clip_idx=None if token_text.strip() else int(clip_idx),
                    scene=scene, reward=reward, log_name=token_to_cache_path[token].parent.name)
    metrics = [k for k in SCALAR_REWARD_KEYS + ARRAY_REWARD_KEYS if k in reward]

    info = f"Token: {token}\nLog: {_current['log_name']}\nVocab size: {len(vocab_8)}"
    bev, info = _finish(bev_canvas(scene, show_gt)[0], dpi, compact, info, {})
    return bev, info, gr.update(choices=metrics, value=metrics[0] if metrics else None)


def render_callback(metric, top_k, prefer, dpi, traj_lw, show_gt, show_markers,
                    compact, color_mode, traj_only):
    if "reward" not in _current:
        return None, "No scene loaded. Load a scene first."
    reward = _current["reward"]
    if metric not in reward:
        return None, f"Reward metric '{metric}' not found in cache."

    fig, ax = bev_canvas(_current["scene"], show_gt, traj_only)
    prefer_safe = prefer == "safest"
    if metric == "ego_areas":
        info = render_ego_areas(ax, reward[metric], int(top_k), prefer_safe, float(traj_lw), show_markers)
    elif metric in ARRAY_REWARD_KEYS:
        info = render_ttc(ax, reward[metric], int(top_k), prefer_safe, float(traj_lw), show_markers)
    else:
        info = render_scalar(ax, reward[metric], int(top_k), color_mode, float(traj_lw), show_markers)

    return _finish(fig, dpi, compact, f"Metric: {metric}\n{info}", {
        "reward_metric": metric, "top_k": int(top_k), "prefer": prefer, "color_mode": color_mode,
    })


def sample_callback(sampling_num, alpha, dpi, traj_lw, show_gt, show_markers,
                    compact, color_mode, traj_only):
    """Sample trajectories from the vocab as in training, colored by pdm_score."""
    if "reward" not in _current:
        return None, "No scene loaded. Load a scene first."
    reward = _current["reward"]
    indices = sample_trajs(reward, int(sampling_num), float(alpha))
    _current["sampled_trajs"] = vocab_8[indices]

    scores = np.asarray(reward["pdm_score"])[indices]
    fig, ax = bev_canvas(_current["scene"], show_gt, traj_only)
    draw_scored_trajs(ax, vocab_8[indices], scores, color_mode=color_mode,
                      lw=float(traj_lw), show_markers=show_markers)

    info = "\n".join([
        f"Sampled {len(indices)} trajectories (alpha = {alpha})",
        f"Unique vocab indices: {len(np.unique(indices))} / {len(indices)}",
        f"pdm_score range: [{scores.min():.4f}, {scores.max():.4f}]  Mean: {scores.mean():.4f}",
    ])
    return _finish(fig, dpi, compact, info, {
        "sampling_num": int(sampling_num), "sampling_alpha": float(alpha), "color_mode": color_mode,
    })


def noise_callback(t, seed, dpi, traj_lw, show_gt, show_markers, compact, traj_only):
    """Apply flow-matching noise at timestep t to the last sampled trajectories."""
    if "sampled_trajs" not in _current:
        return None, "No sampled trajectories. Run 'Sample Trajectories' first."
    clean = _current["sampled_trajs"]
    seed = int(seed) if seed is not None else None
    noisy = add_flow_noise(clean, float(t), seed=seed)

    fig, ax = bev_canvas(_current["scene"], show_gt, traj_only)
    for traj in noisy:
        poses = np.vstack([[0, 0, 0], traj])
        ax.plot(poses[:, 1], poses[:, 0], color=NEW_TAB_10[0], linewidth=float(traj_lw), alpha=0.6, zorder=10)
        if show_markers:
            draw_time_markers(ax, traj)

    disp = np.linalg.norm(noisy[..., :2] - clean[..., :2], axis=-1)
    info = "\n".join([
        f"Flow matching noise: t = {float(t):.2f}, seed = {seed}",
        f"Trajectories: {len(clean)}",
        f"Displacement mean / max: {disp.mean():.3f}m / {disp.max():.3f}m",
    ])
    return _finish(fig, dpi, compact, info, {"noise_t": float(t), "noise_seed": seed})


def show_features_callback(dpi):
    if "token" not in _current:
        return None, None, "No scene loaded."
    return render_cached_features(token_to_cache_path[_current["token"]], dpi=int(dpi))


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="FlowR2A Train Data Inspector") as app:
        gr.Markdown("# FlowR2A Train Data Inspector")
        gr.Markdown(f"Tokens: {len(token_to_cache_path)} | Clips: {len(clips)} | Vocab: {len(vocab_8)}")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Scene")
                token_input = gr.Textbox(label="Token (overrides clip selection)",
                                         placeholder="e.g. 0a678d2136b35b56")
                clip_slider = gr.Slider(0, max(len(clips) - 1, 0), step=1, value=0, label="Clip index")
                clip_info_text = gr.Textbox(label="Clip info", interactive=False, lines=2)
                frame_slider = gr.Slider(0, 50, step=1, value=0, label="Frame in clip")
                load_btn = gr.Button("Load Scene", variant="primary")
                scene_info = gr.Textbox(label="Scene info", interactive=False, lines=3)

            with gr.Column(scale=1):
                gr.Markdown("### Reward Visualization")
                reward_metric = gr.Dropdown(choices=[], value=None, label="Reward metric", interactive=True)
                top_k = gr.Slider(0, 8192, step=1, value=50, label="Number of trajectories to show (0 = all)",
                                  info="Scalar metrics show the top-k by reward")
                prefer = gr.Radio(["most violations", "safest"], value="most violations",
                                  label="Per-step metrics (ego_areas / ttc_time): show first")
                render_btn = gr.Button("Render", variant="primary")

        with gr.Accordion("Training sampling & flow-matching noise", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling_num = gr.Slider(1, 500, step=1, value=50, label="Number of samples")
                    sampling_alpha = gr.Slider(0.0, 2.0, step=0.05, value=SAMPLING_ALPHA,
                                               label="Alpha (weights = (1 / prob_density) ** alpha)")
                    sample_btn = gr.Button("Sample Trajectories", variant="primary")
                with gr.Column(scale=1):
                    t_slider = gr.Slider(0.0, 1.0, step=0.01, value=1.0,
                                         label="Timestep t (1 = clean, 0 = pure noise)")
                    noise_seed = gr.Number(value=42, label="Noise seed", precision=0)
                    noise_btn = gr.Button("Apply Noise & Render", variant="primary")

        with gr.Accordion("Render options", open=False):
            with gr.Row():
                dpi_slider = gr.Slider(80, 300, step=10, value=120, label="DPI")
                traj_lw_slider = gr.Slider(0.2, 5.0, step=0.1, value=1.0, label="Line width")
                color_mode = gr.Radio(["absolute", "minmax"], value="absolute", label="Score color range",
                                      info="absolute: [0, 1]; minmax: stretch to this scene's min/max score")
            with gr.Row():
                show_gt = gr.Checkbox(value=True, label="Show GT trajectory")
                show_markers = gr.Checkbox(value=False, label="Waypoint markers (colored by time)")
                traj_only = gr.Checkbox(value=False, label="Trajectories only (no map)")
                compact_mode = gr.Checkbox(value=False, label="Compact (no margin)")

        with gr.Row():
            bev_image = gr.Image(label="BEV (scalar rewards colored with inferno)", type="pil")
            with gr.Column():
                result_text = gr.Textbox(label="Results", lines=10, interactive=False)
                with gr.Row():
                    save_btn = gr.Button("Save Image + Config")
                    save_status = gr.Textbox(label="Save status", interactive=False, lines=1)

        with gr.Accordion("Cached features (LiDAR / camera)", open=False):
            show_features_btn = gr.Button("Show Features")
            with gr.Row():
                lidar_image = gr.Image(label="LiDAR BEV", type="pil")
                camera_image = gr.Image(label="Camera (stitched)", type="pil")
            feature_info_text = gr.Textbox(label="Feature info", interactive=False, lines=3)

        render_opts = [dpi_slider, traj_lw_slider, show_gt, show_markers, compact_mode]
        clip_slider.change(lambda i: clip_info(clips, i), inputs=[clip_slider],
                           outputs=[clip_info_text, frame_slider])
        load_btn.click(load_scene_callback,
                       inputs=[token_input, clip_slider, frame_slider, dpi_slider, show_gt, compact_mode],
                       outputs=[bev_image, scene_info, reward_metric])
        render_btn.click(render_callback,
                         inputs=[reward_metric, top_k, prefer, *render_opts, color_mode, traj_only],
                         outputs=[bev_image, result_text])
        sample_btn.click(sample_callback,
                         inputs=[sampling_num, sampling_alpha, *render_opts, color_mode, traj_only],
                         outputs=[bev_image, result_text])
        noise_btn.click(noise_callback,
                        inputs=[t_slider, noise_seed, *render_opts, traj_only],
                        outputs=[bev_image, result_text])
        save_btn.click(lambda: save_result(SAVE_DIR, _current), outputs=[save_status])
        show_features_btn.click(show_features_callback, inputs=[dpi_slider],
                                outputs=[lidar_image, camera_image, feature_info_text])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global vocab_8, token_to_cache_path, clips

    parser = argparse.ArgumentParser(description="Interactive FlowR2A train data inspector")
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH,
                        help="Training cache directory (with transfuser_reward.gz per token)")
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH,
                        help="Vocabulary trajectories .npy path")
    parser.add_argument("--clip-index", type=str, default="",
                        help="Pre-built clip index pkl from analyze_scene_clips.py (optional)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    vocab_8 = np.load(args.vocab_path)[:, DOWNSAMPLE_INDICES, :]  # (N, 40, 3) -> (N, 8, 3)
    print(f"Vocabulary: {len(vocab_8)} trajectories from {args.vocab_path}")

    token_to_cache_path = build_token_index(args.cache_path)
    if args.clip_index and os.path.exists(args.clip_index):
        clips = load_clip_index(args.clip_index, token_to_cache_path)
        print(f"Loaded {len(clips)} clips from {args.clip_index}")
    else:
        clips = build_fallback_clips(token_to_cache_path)
        print(f"Built {len(clips)} clips (grouped by scene directory)")

    build_app().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
