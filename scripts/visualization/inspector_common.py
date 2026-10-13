"""Shared helpers for the FlowR2A Gradio inspectors.

Used by interactive_inspector.py (agent inference on navtest) and
interactive_train_inspector.py (navtrain reward cache).
"""

import io
import json
import pickle
from pathlib import Path
from typing import Any, Container, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image

from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

# Trajectories are colored by score (low = dark, high = bright)
SCORE_CMAP = plt.get_cmap("inferno")
# Waypoint markers are colored by time order (dark = early, bright = late)
TIME_COLORS = [plt.get_cmap("viridis")(i / 7) for i in range(8)]
MARKER_SIZE = 30


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def fig_to_pil(fig, dpi=120, compact=False) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0 if compact else 0.1)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def draw_gt(ax, scene):
    gt = scene.get_future_trajectory(num_trajectory_frames=8)
    add_trajectory_to_bev_ax(ax, gt, TRAJECTORY_CONFIG["human"])


def draw_time_markers(ax, traj):
    """Draw the waypoints of an (8, 3) trajectory, colored by time order."""
    ax.scatter(traj[:, 1], traj[:, 0], c=TIME_COLORS[:len(traj)], s=MARKER_SIZE,
               zorder=12, edgecolors="black", linewidths=0.3)


def draw_scored_trajs(ax, trajs, scores, top_k=0, color_mode="absolute",
                      lw=1.0, show_markers=False) -> np.ndarray:
    """Draw (K, 8, 3) trajectories colored by score, best on top.

    NaN scores are skipped and top_k > 0 keeps only the k highest. color_mode
    "absolute" maps scores in [0, 1] to the colormap; "minmax" stretches it to
    the range of valid scores. Returns the drawn indices (ascending score).
    """
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.where(~np.isnan(scores))[0]
    order = valid[np.argsort(scores[valid])]
    if 0 < top_k < len(order):
        order = order[-top_k:]
    if len(order) == 0:
        return order

    lo, hi = (scores[valid].min(), scores[valid].max()) if color_mode == "minmax" else (0.0, 1.0)
    for idx in order:
        norm = 0.5 if hi - lo < 1e-6 else (scores[idx] - lo) / (hi - lo)
        poses = np.vstack([[0, 0, 0], trajs[idx]])
        ax.plot(poses[:, 1], poses[:, 0], color=SCORE_CMAP(np.clip(norm, 0, 1)),
                linewidth=lw, alpha=0.6, zorder=10)
        if show_markers:
            draw_time_markers(ax, trajs[idx])
    return order


def render_cached_features(token_path: Path, dpi=120) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """Render a token's cached transfuser_feature.gz: LiDAR BEV per frame + camera.

    The LiDAR feature is (C, 256, 256) with C = 4 frames x (below, above ground),
    or 4 x (above) when the ground plane is not split.
    """
    feature_path = token_path / "transfuser_feature.gz"
    if not feature_path.exists():
        return None, None, f"Feature cache not found: {feature_path}"
    data = {k: v.numpy() if isinstance(v, torch.Tensor) else v
            for k, v in load_feature_target_from_pickle(feature_path).items()}
    lidar_img = camera_img = None
    info = [f"Feature cache: {token_path}"]

    if "camera_feature" in data:
        cam = data["camera_feature"]
        camera_img = Image.fromarray((np.clip(cam.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
        info.append(f"Camera: {cam.shape}")

    if "lidar_feature" in data:
        lidar = data["lidar_feature"]
        info.append(f"LiDAR: {lidar.shape}")
        num_frames = 4
        layers = [("below", "#4e79a7"), ("above", "#e15759")]
        if lidar.shape[0] // num_frames == 1:
            layers = layers[1:]

        fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
        for f, ax in enumerate(axes):
            counts = []
            for l, (name, color) in enumerate(layers):
                bev = lidar[f * len(layers) + l]
                rows, cols = np.nonzero(bev > 0)
                if len(rows):
                    ax.scatter(cols, rows, c=color, s=1.5, alpha=np.clip(bev[rows, cols], 0.2, 1.0),
                               edgecolors="none", zorder=l + 1)
                counts.append(f"{name}:{len(rows)}")
            label = "t (current)" if f == num_frames - 1 else f"t-{num_frames - 1 - f}"
            ax.set_title(f"{label}\n{' '.join(counts)}", fontsize=9)
            ax.set_xlim(0, lidar.shape[2])
            ax.set_ylim(0, lidar.shape[1])
            ax.set_aspect("equal")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[-1].legend(handles=[Patch(facecolor=c, label=f"{n.capitalize()} ground") for n, c in layers],
                        loc="lower right", fontsize=7)
        fig.suptitle("LiDAR BEV per frame ([-32m, 32m])", fontsize=12)
        fig.tight_layout()
        lidar_img = fig_to_pil(fig, dpi=dpi)

    if "status_feature" in data:
        info.append(f"Status: {data['status_feature'].shape}")
    return lidar_img, camera_img, "\n".join(info)


# ---------------------------------------------------------------------------
# Clips, token selection, saving
# ---------------------------------------------------------------------------

def sort_clips(clips: List[Dict]) -> List[Dict]:
    """Drop empty clips, refresh num_frames, and sort longest first."""
    clips = [{**c, "num_frames": len(c["tokens"])} for c in clips if c["tokens"]]
    return sorted(clips, key=lambda c: c["num_frames"], reverse=True)


def load_clip_index(path: str, valid_tokens: Container[str]) -> List[Dict]:
    """Load a clip index from analyze_scene_clips.py, keeping only valid tokens."""
    with open(path, "rb") as f:
        clips = pickle.load(f)["clips"]
    return sort_clips([{**c, "tokens": [t for t in c["tokens"] if t in valid_tokens]} for c in clips])


def resolve_token(token_text: str, clip_idx, frame_idx, clips: List[Dict],
                  valid_tokens: Container[str]) -> Optional[str]:
    """Return the typed token if given, else frame `frame_idx` of clip `clip_idx`."""
    token_text = token_text.strip()
    if token_text:
        return token_text if token_text in valid_tokens else None
    clip_idx, frame_idx = int(clip_idx), int(frame_idx)
    if not 0 <= clip_idx < len(clips) or not 0 <= frame_idx < len(clips[clip_idx]["tokens"]):
        return None
    return clips[clip_idx]["tokens"][frame_idx]


def clip_info(clips: List[Dict], clip_idx):
    """Clip summary text and frame-slider update for the clip slider callback."""
    clip_idx = int(clip_idx)
    if not 0 <= clip_idx < len(clips):
        return "No clips loaded", gr.update(maximum=0, value=0)
    clip = clips[clip_idx]
    info = f"Log: {clip['log_name']}\nFrames: {clip['num_frames']}"
    return info, gr.update(maximum=max(clip["num_frames"] - 1, 0), value=0)


def save_result(out_root: str, current: Dict[str, Any]) -> str:
    """Save current["last_bev"] as PNG and its settings/info as JSON next to it.

    Files go to <out_root>/clip_<idx>/ (or direct/ when the token was typed in)
    as <token>_<n>.png/.json, with n incremented to avoid overwriting.
    """
    if "last_bev" not in current:
        return "Nothing to save yet."
    token, clip_idx = current["token"], current.get("clip_idx")
    out_dir = Path(out_root) / (f"clip_{clip_idx}" if clip_idx is not None else "direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while (out_dir / f"{token}_{i}.png").exists():
        i += 1
    img_path = out_dir / f"{token}_{i}.png"
    current["last_bev"].save(img_path)
    meta = {"token": token, "clip_idx": clip_idx, **current.get("last_settings", {}),
            "info": current.get("last_info", "")}
    with open(img_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return f"Saved: {img_path}"
