# Reward Simulation

This guide generates the per-token reward cache (`transfuser_reward.gz`) from scratch. If you only want to train, it's faster to download the pre-computed cache (see [Training](training.md#trajectory-vocabulary-and-reward-cache)).

Each token's reward cache has:
- PDM sub-scores for all 8192 vocab trajectories, taken from the GTRS reward pack.
- Two derived keys: `prob_density`, a KDE density of the per-token `pdm_score` distribution, and `safe_ego_progress_normed`, which is ego progress with TTC-failing trajectories set to zero, normalized by the per-token max.
- Re-simulated PDM sub-scores (`*_v1`) and the 40-step ego-area / TTC ground truth (`ego_areas`, `ttc_time`).

## 1. Prerequisites
Finish these steps from [Training](training.md) first:
- `ckpts/8192.npy`: the trajectory vocabulary.
- Feature cache at `$NAVSIM_EXP_ROOT/training_cache`. The reward files are written into its `<scene>/<token>/` dirs.
- Metric cache at `$NAVSIM_EXP_ROOT/train_pdm_cache`. It's used for the PDM simulation.

## 2. Download the reward pack
Download `navtrain_8192.pkl` (~15 GB) from the [🤗 GTRS model page](https://huggingface.co/Zzxxxxxxxx/gtrs). It holds the PDM sub-scores of the 8192 vocab trajectories for every navtrain token.

```bash
huggingface-cli download Zzxxxxxxxx/gtrs navtrain_8192.pkl --local-dir ckpts
```

## 3. Simulate and cache rewards
`run_train_reward_caching.sh` runs `scripts/caching/run_train_reward_caching.py`. For each token, the script:
1. Adds `prob_density` and `safe_ego_progress_normed` to the reward-pack entry if they're missing. It also zeroes `ego_progress` for trajectories that fail TTC, which is the target for the stage-2 EP head.
2. Runs the full PDM simulation (TTC bound 2.0 s) for all 8192 trajectories. It saves the sub-scores with a `_v1` suffix, plus `ego_areas` and `ttc_time`.
3. Writes `<training_cache>/<scene>/<token>/transfuser_reward.gz`.

The work is split into chunks, so you can spread it across processes or machines:

```bash
# bash scripts/caching/run_train_reward_caching.sh [CHUNK_ID] [TOTAL_CHUNKS]
bash scripts/caching/run_train_reward_caching.sh 0 4
bash scripts/caching/run_train_reward_caching.sh 1 4
bash scripts/caching/run_train_reward_caching.sh 2 4
bash scripts/caching/run_train_reward_caching.sh 3 4
```

Each run uses 64 workers (`--num-workers` in the `.sh`). It loads the full reward pack into memory, so plan for at least 15 GB of RAM per running chunk. `--skip-existing` is on, so an interrupted run picks up where it stopped. This step runs a full PDM simulation for every token and takes a while; plan for it to run for many hours on a single machine.

## 4. Validate
Check that every reward file loads and has all the expected keys:

```bash
python scripts/caching/check_reward.py \
    --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
    --num-workers 64 --out bad_reward_files.json
```

If any files are reported as corrupt or missing keys, delete them and re-run step 3. `--skip-existing` makes step 3 regenerate only the missing files.

When validation passes, the cache is ready for training. `reward_cache_path` in the training configs points to the feature cache dir by default.

## 5. (Optional) Match the paper's TTC ground truth
The paper's reward cache also stores `ttc_time_b15`, the per-step TTC simulated with a 1.5 s bound. When it's present, the model uses it in place of `ttc_time`. Since `ttc_time` is clipped to 1.5 s during training, the 2.0 s simulation from step 3 gives nearly the same result, so you can skip this step. To match the paper exactly, run `patch_reward.py`. It re-runs the simulation at the 1.5 s bound and adds `ttc_time_b15` to each existing `transfuser_reward.gz` in place:

```bash
python scripts/caching/patch_reward.py \
    --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
    --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
    --num-workers 64 --skip-existing --chunks 0/4   # repeat for 1/4, 2/4, 3/4
```

This step runs another full PDM simulation for every token, so it takes about as long as step 3.
