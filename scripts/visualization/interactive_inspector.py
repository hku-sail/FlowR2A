#!/usr/bin/env python3
"""Interactive Gradio inspector for FlowR2A inference on navtest.

Loads a trained agent, runs it on a single navtest token and draws all sampled
trajectory proposals on BEV, colored by the scorer's predicted score. Sampling
settings (number of proposals, reward condition, scorer weights) can be changed
between runs.

Usage:
    python scripts/visualization/interactive_inspector.py \
        --checkpoint /path/to/epoch=...ckpt --port 7860

`--agent_config` is the Hydra agent config name (default flowr2a_agent_s2).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.visualization.plots import plot_bev_frame

from inspector_common import (
    clip_info, draw_gt, draw_scored_trajs, fig_to_pil, load_clip_index,
    render_cached_features, resolve_token, save_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ROOT = os.environ["NAVSIM_EXP_ROOT"]
FEATURE_CACHE = os.path.join(EXP_ROOT, "testing_cache")
CLIP_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navtest_clips_index.pkl")
DEFAULT_AGENT_CONFIG = "flowr2a_agent_s2"
SAVE_DIR = "viz_output/inspector"

OPENSCENE_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
NAVSIM_LOG_PATH = OPENSCENE_ROOT / "navsim_logs" / "test"
SENSOR_BLOBS_PATH = OPENSCENE_ROOT / "sensor_blobs" / "test"

# Puts the model in eval mode so forward() returns all proposals and scores
os.environ["EVAL"] = "1"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
agent: Optional[AbstractAgent] = None
dataset: Optional[CacheOnlyDataset] = None
clips: List[Dict] = []
_current: Dict[str, Any] = {}
_agent_info: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_agent(checkpoint: str, agent_config: str):
    """Build the agent from its Hydra config, load the checkpoint, and index the navtest cache."""
    global agent, dataset, clips

    config_dir = os.path.join(
        os.environ.get("NAVSIM_DEVKIT_ROOT", "."),
        "navsim/planning/script/config/pdm_scoring",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            f"agent={agent_config}",
            f"agent.checkpoint_path='{checkpoint}'",
        ])

    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.cuda()
    _agent_info.update(checkpoint=checkpoint, agent_config=agent_config)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    dataset = CacheOnlyDataset(
        cache_path=FEATURE_CACHE,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=scene_filter.log_names,
        train=False,
    )
    if os.path.exists(CLIP_INDEX):
        clips = load_clip_index(CLIP_INDEX, dataset._valid_cache_paths)
    print(f"Loaded agent {agent_config} | {len(dataset.tokens)} tokens | {len(clips)} clips")


def load_scene_for_token(token: str) -> Any:
    log_name = dataset._valid_cache_paths[token].parent.name
    sf = SceneFilter(
        tokens=[token], log_names=[log_name],
        num_history_frames=4, num_future_frames=10, frame_interval=1,
    )
    sl = SceneLoader(
        data_path=NAVSIM_LOG_PATH, sensor_blobs_path=SENSOR_BLOBS_PATH,
        scene_filter=sf, sensor_config=SensorConfig.build_no_sensors(),
    )
    return sl.get_scene_from_token(token)


def batch_single(features, targets):
    """Add a batch dim to one sample's tensors and move them to GPU."""
    def _batch(d, wrap=()):
        return {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else ([v] if k in wrap else v)
                for k, v in d.items()}
    return _batch(features, wrap=("token",)), _batch(targets)


def bev_canvas(scene, show_gt):
    fig, ax = plot_bev_frame(scene, scene.scene_metadata.num_history_frames - 1)
    if show_gt:
        draw_gt(ax, scene)
    return fig, ax


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def load_scene_callback(token_text, clip_idx, frame_idx, dpi, show_gt, compact):
    token = resolve_token(token_text, clip_idx, frame_idx, clips, dataset._valid_cache_paths)
    if token is None:
        return None, "Token not found in dataset."

    scene = load_scene_for_token(token)
    features, targets = dataset._load_scene_with_token(token)
    log_name = dataset._valid_cache_paths[token].parent.name

    bev = fig_to_pil(bev_canvas(scene, show_gt)[0], dpi=int(dpi), compact=compact)
    info = f"Token: {token}\nLog: {log_name}"
    _current.clear()
    _current.update(
        token=token, clip_idx=None if token_text.strip() else int(clip_idx),
        scene=scene, features=features, targets=targets,
        last_bev=bev, last_info=info, last_settings={"log_name": log_name},
    )
    return bev, info


def run_inference_callback(num_traj, score_min, score_max,
                           init_step_min, init_step_max, w_ttc, w_ep, w_c,
                           top_k, dpi, traj_lw, show_gt, show_markers, compact, color_mode):
    if "features" not in _current:
        return None, "No scene loaded. Load a scene first."

    settings = {
        "test_num_traj_sampling": int(num_traj),
        "test_score_min": float(score_min),
        "test_score_max": float(score_max),
        "init_step_min": int(init_step_min),
        "init_step_max": int(init_step_max),
        "test_weight_ttc": float(w_ttc),
        "test_weight_ep": float(w_ep),
        "test_weight_c": float(w_c),
    }
    head = agent._transfuser_model._trajectory_head
    for name, value in settings.items():
        setattr(head, name, value)

    bf, bt = batch_single(_current["features"], _current["targets"])
    with torch.no_grad():
        out = agent.forward(bf, bt, multi_return=True)

    # scores = NC * DAC * weighted mean of (TTC, EP, C), using the weights above
    trajs = out["trajs"][0].cpu().numpy()
    scores = out["scores"][0].cpu().numpy()
    best = int(out["best_idx"][0].item())

    lines = [
        f"Trajectories: {len(trajs)}",
        f"Best idx: {best}  Score: {scores[best]:.4f}",
        f"Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]  Mean: {np.nanmean(scores):.4f}",
        "",
        "Subscores (best traj):",
    ]
    lines += [f"  {name}: {s[0, best].item():.4f}" for name, s in out["subscore"].items()]

    fig, ax = bev_canvas(_current["scene"], show_gt)
    draw_scored_trajs(ax, trajs, scores, top_k=int(top_k), color_mode=color_mode,
                      lw=float(traj_lw), show_markers=show_markers)
    bev = fig_to_pil(fig, dpi=int(dpi), compact=compact)

    _current["last_bev"] = bev
    _current["last_info"] = "\n".join(lines)
    _current["last_settings"].update(_agent_info, **settings, top_k=int(top_k))
    return bev, "\n".join(lines)


def show_features_callback(dpi):
    if "token" not in _current:
        return None, None, "No scene loaded."
    return render_cached_features(dataset._valid_cache_paths[_current["token"]], dpi=int(dpi))


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app():
    head = agent._transfuser_model._trajectory_head
    with gr.Blocks(title="FlowR2A Inspector") as app:
        gr.Markdown("# FlowR2A Inspector")
        gr.Markdown(f"Agent: `{_agent_info['agent_config']}` | "
                    f"Checkpoint: `{os.path.basename(_agent_info['checkpoint'])}` | "
                    f"Tokens: {len(dataset.tokens)} | Clips: {len(clips)}")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Scene")
                token_input = gr.Textbox(label="Token (overrides clip selection)",
                                         placeholder="e.g. 67628d15c5b45860")
                clip_slider = gr.Slider(0, max(len(clips) - 1, 0), step=1, value=0, label="Clip index")
                clip_info_text = gr.Textbox(label="Clip info", interactive=False, lines=2)
                frame_slider = gr.Slider(0, 50, step=1, value=0, label="Frame in clip")
                load_btn = gr.Button("Load Scene", variant="primary")
                scene_info = gr.Textbox(label="Scene info", interactive=False, lines=2)

            with gr.Column(scale=1):
                gr.Markdown("### Inference")
                num_traj = gr.Slider(1, 200, step=1, value=head.test_num_traj_sampling,
                                     label="Number of sampled trajectories")
                with gr.Row():
                    score_min = gr.Slider(0, 1, step=0.01, value=head.test_score_min, label="Reward condition min")
                    score_max = gr.Slider(0, 1, step=0.01, value=head.test_score_max, label="Reward condition max")
                top_k = gr.Slider(0, 200, step=1, value=0, label="Show top-k by score (0 = all)")
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        init_step_min = gr.Slider(0, 20, step=1, value=head.init_step_min, label="init_step_min")
                        init_step_max = gr.Slider(0, 20, step=1, value=head.init_step_max, label="init_step_max")
                    with gr.Row():
                        w_ttc = gr.Slider(0, 5, step=0.1, value=head.test_weight_ttc, label="Scorer weight TTC")
                        w_ep = gr.Slider(0, 5, step=0.1, value=head.test_weight_ep, label="Scorer weight EP")
                        w_c = gr.Slider(0, 5, step=0.1, value=head.test_weight_c, label="Scorer weight C")
                run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Accordion("Render options", open=False):
            with gr.Row():
                dpi_slider = gr.Slider(80, 300, step=10, value=120, label="DPI")
                traj_lw_slider = gr.Slider(0.2, 5.0, step=0.1, value=1.0, label="Line width")
                color_mode = gr.Radio(["absolute", "minmax"], value="absolute", label="Score color range",
                                      info="absolute: [0, 1]; minmax: stretch to this scene's min/max score")
            with gr.Row():
                show_gt = gr.Checkbox(value=True, label="Show GT trajectory")
                show_markers = gr.Checkbox(value=False, label="Waypoint markers (colored by time)")
                compact_mode = gr.Checkbox(value=False, label="Compact (no margin)")

        with gr.Row():
            bev_image = gr.Image(label="BEV (trajectories colored by score, inferno)", type="pil")
            with gr.Column():
                result_text = gr.Textbox(label="Results", lines=12, interactive=False)
                with gr.Row():
                    save_btn = gr.Button("Save Image + Config")
                    save_status = gr.Textbox(label="Save status", interactive=False, lines=1)

        with gr.Accordion("Cached features (LiDAR / camera)", open=False):
            show_features_btn = gr.Button("Show Features")
            with gr.Row():
                lidar_image = gr.Image(label="LiDAR BEV", type="pil")
                camera_image = gr.Image(label="Camera (stitched)", type="pil")
            feature_info_text = gr.Textbox(label="Feature info", interactive=False, lines=3)

        clip_slider.change(lambda i: clip_info(clips, i), inputs=[clip_slider],
                           outputs=[clip_info_text, frame_slider])
        load_btn.click(load_scene_callback,
                       inputs=[token_input, clip_slider, frame_slider, dpi_slider, show_gt, compact_mode],
                       outputs=[bev_image, scene_info])
        run_btn.click(run_inference_callback,
                      inputs=[num_traj, score_min, score_max,
                              init_step_min, init_step_max, w_ttc, w_ep, w_c,
                              top_k, dpi_slider, traj_lw_slider, show_gt, show_markers,
                              compact_mode, color_mode],
                      outputs=[bev_image, result_text])
        save_btn.click(lambda: save_result(SAVE_DIR, _current), outputs=[save_status])
        show_features_btn.click(show_features_callback, inputs=[dpi_slider],
                                outputs=[lidar_image, camera_image, feature_info_text])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive FlowR2A inference inspector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a FlowR2A checkpoint (.ckpt)")
    parser.add_argument("--agent_config", type=str, default=DEFAULT_AGENT_CONFIG, help="Hydra agent config name")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_agent(args.checkpoint, args.agent_config)
    build_app().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
