# Training

This guide trains FlowR2A on the NAVSIM **navtrain** split. Make sure you have finished [Data and Environment Preparation](install.md) first. Training runs in two stages: stage 1 trains the flow decoder, stage 2 trains the scorer on top of a stage-1 checkpoint.

## 1. Download

### Training data
Training uses the **navtrain** split (nuPlan maps + the OpenScene `trainval` sensor blobs and logs). A download script is provided in `download/`:

```bash
cd download
bash download_maps.sh        # skip if you already have maps from evaluation
bash download_navtrain.sh
cd ..
```

Arrange the data next to the evaluation data so both splits are discoverable:
```
<WORKSPACE>/navsim_workspace/dataset
├── maps
├── navsim_logs
│   ├── test
│   └── trainval
└── sensor_blobs
    ├── test
    └── trainval
```

### Trajectory vocabulary and reward cache
Both are hosted on the [🤗 Hugging Face model page](https://huggingface.co/lixirui142/FlowR2A). Download them into `ckpts/`:

```bash
# Vocab trajectories (8192.npy) — required for training-time sampling.
huggingface-cli download lixirui142/FlowR2A 8192.npy --local-dir ckpts

# Reward cache — released as split part files (reward_cache_part_*.pkl).
# Computing rewards from scratch runs full PDM simulation for all 8192 vocab
# trajectories per token, which is slow, so we release them pre-computed.
# To generate them yourself, see docs/reward_simulation.md.
huggingface-cli download lixirui142/FlowR2A --include 'reward_cache/*' --local-dir $NAVSIM_EXP_ROOT
```

The config reads the vocab from `ckpts/8192.npy` (`traj_list_path` in `navsim/agents/flowr2a/flowr2a_config.py`).

### Backbone weights (optional)
The ResNet-34 backbone is downloaded automatically from `timm` on first use. If it fails, you can manually download [🤗 timm/resnet34.a1_in1k](https://huggingface.co/timm/resnet34.a1_in1k) into `ckpts/`:

```bash
huggingface-cli download timm/resnet34.a1_in1k pytorch_model.bin --local-dir ckpts/resnet34.a1_in1k
```

The config loads `ckpts/resnet34.a1_in1k/pytorch_model.bin` if present (`bkb_path` in `flowr2a_config.py`), when timm download fails.


## 2. Caching

### Feature cache
Pre-compute the per-token feature/target cache for navtrain with `run_dataset_caching.py`. The training configs expect it at `$NAVSIM_EXP_ROOT/training_cache` (`cache_path` in `navsim/planning/script/config/training/flowr2a_training_s{1,2}.yaml`):

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=flowr2a_agent_s1 \
    experiment_name=flowr2a_train_cache \
    train_test_split=navtrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache
```

This writes `<cache>/<scene>/<token>/transfuser_feature.gz` and `transfuser_target.gz` for every token.

### Reward cache
Unpack the downloaded reward parts into the feature cache so each token gets its `transfuser_reward.gz`:

```bash
python scripts/caching/reward_cache_pack.py extract \
    --pack-dir $NAVSIM_EXP_ROOT/reward_cache \
    --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
    --num-workers 64 --skip-existing
```

The extractor matches each token in the pack to its scene dir in the feature cache and writes `<cache>/<scene>/<token>/transfuser_reward.gz`. `reward_cache_path` in the training configs points at the same dir as `cache_path` by default.

### Metric cache (stage 2 only)
Stage 2 runs online PDM scoring during training. The metric cache speeds up the online PDM scoring. The model reads it from `$NAVSIM_EXP_ROOT/train_pdm_cache` with layout `<scene>/unknown/<token>/metric_cache.pkl` (`pdm_cache_dir` in `flowr2a_config.py`):

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain \
    cache.cache_path=$NAVSIM_EXP_ROOT/train_pdm_cache
```

Stage 1 does not use this cache and can be trained without it.

## 3. Train
The launcher scripts set `TRAIN_CONFIG` and call `run_training.py`. Checkpoints are written under `$NAVSIM_EXP_ROOT/<experiment_name>`.

```bash
# Stage 1 — flow decoder
bash scripts/training/run_s1_train.sh

# Stage 2 — scorer, initialized from the stage-1 checkpoint
bash scripts/training/run_s2_train.sh /path/to/stage1/ckpt
```

Any extra arguments are forwarded to `run_training.py` as Hydra overrides.

## 4. Evaluate
Score a trained checkpoint on navtest (see [Evaluation](evaluation.md) for cache setup):
```bash
python scripts/evaluation/run_eval.py --exp_id flowr2a_agent_s2 --checkpoint /path/to/epoch=...ckpt
```
