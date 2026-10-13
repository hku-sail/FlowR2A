# Visualization

Two Gradio apps are provided for interactive inspection in the browser:

- `scripts/visualization/interactive_inspector.py` runs a trained FlowR2A agent on navtest scenes and shows all sampled trajectory proposals, colored by the scorer's predicted score.
- `scripts/visualization/interactive_train_inspector.py` browses the navtrain reward cache: rewards of all 8192 vocabulary trajectories per scene, plus a preview of the training-time trajectory sampling and flow-matching noise. No model is needed.

## 1. Install dependencies
```bash
pip install gradio==4.44.1 diffusers==0.31.0 "pydantic==2.10.6" huggingface_hub==0.36.2
```

## 2. Inference inspector
This needs the checkpoint and the navtest feature cache from [Evaluation](evaluation.md) (steps 1–2), and the OpenScene `test` logs from [Data and Environment Preparation](install.md).

```bash
python scripts/visualization/interactive_inspector.py \
    --checkpoint ckpts/flowr2a_s2.ckpt \
    --port 7860
```

Open `http://localhost:7860`, then:

1. **Scene**: pick a clip and a frame in it, or paste a token directly, then click **Load Scene**. Clips are continuous frame sequences from the bundled `scripts/visualization/navtest_clips_index.pkl`.
2. **Inference**: set the number of sampled trajectories and the reward condition range (the target score that guides generation), then click **Run Inference**. The BEV shows every proposal colored by predicted score (inferno, dark = low, bright = high), the GT trajectory, and the best proposal's sub-scores (NC, DAC, TTC, EP, C).
3. **Advanced** exposes the flow init-step range and the scorer weights for TTC / EP / C. Defaults match `navsim/agents/flowr2a/flowr2a_config.py`.

Proposals for a single scene often score in a narrow range. Set **Render options → Score color range** to `minmax` to spread the colormap over that range.

## 3. Reward cache inspector
This needs the navtrain feature cache with unpacked rewards (`transfuser_reward.gz` per token) and `ckpts/8192.npy`, both from [Training](training.md) (steps 1–2), and the OpenScene `trainval` logs.

```bash
python scripts/visualization/interactive_train_inspector.py \
    --cache-path $NAVSIM_EXP_ROOT/training_cache \
    --vocab-path ckpts/8192.npy \
    --port 7860
```

The first launch scans the cache for tokens and saves the index to `viz_output/token_index_<hash>.pkl`, so later launches start quickly. By default, tokens are grouped into clips by scene directory. 

In the app:

1. **Load Scene**, then choose a **Reward metric** and click **Render**.
   - Scalar rewards (`pdm_score`, sub-scores, `prob_density`, ...) color the top-k trajectories by value.
   - `ego_areas` colors each segment by drivable-area / oncoming-lane violation.
   - `ttc_time` (and `ttc_time_b15`, if present) colors each segment by time-to-collision. **Show first** selects whether the most-violating or the safest trajectories are drawn.
2. **Training sampling & flow-matching noise**: **Sample Trajectories** draws vocabulary trajectories with weights `(1 / prob_density) ** alpha`, as in training. **Apply Noise & Render** then noises them to flow timestep `t` (1 = clean, 0 = pure noise).

## Common options
- **Render options** sets DPI, line width, GT trajectory, waypoint markers (colored by time), and compact output. In the reward cache inspector it can also hide the map and show trajectories only.
- **Cached features** shows the LiDAR BEV and camera inputs the model sees for the current token.
- **Save Image + Config** writes the current BEV image and its settings as a PNG and JSON pair to `viz_output/inspector/` or `viz_output/train_inspector/`.
