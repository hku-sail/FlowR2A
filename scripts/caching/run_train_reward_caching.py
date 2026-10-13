"""
Reward caching builder: create transfuser_reward.gz per scene.

The input reward pack is the GTRS `navtrain_8192.pkl` ({token: PDM sub-scores
over the 8192 vocab trajectories}). For each token:
  1. Load the reward-pack entry and add the derived keys if missing:
       - prob_density: KDE density of the vocab's pdm_score distribution.
       - safe_ego_progress_normed: ego_progress with TTC-failing trajectories
         zeroed, normalized by the per-token max.
  2. Keep only the whitelisted keys (_KEEP_KEYS).
  3. Run full PDM simulation for all 8192 vocab trajectories.
  4. Save 40-step iPad GT (ego_areas, ttc_time) at full simulator resolution.
  5. Save the freshly-simulated PDM sub-scores with a _v1 suffix.

Scoring settings: ttc_bound=2.0, use_history_comfort=True,
driving_direction_exclude_intersection=True.

Usage:
    python scripts/caching/run_train_reward_caching.py \
        --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
        --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
        --reward-pack-path ckpts/navtrain_8192.pkl \
        --num-workers 32 --skip-existing
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Add project root to path (scripts/caching/ -> repo root is two levels up)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from navsim.planning.training.dataset import dump_feature_target_to_pickle
from navsim.agents.flowr2a.pdm_scoring import _pdm_worker, _init_pool

# Reward-pack keys to keep in the saved reward dict. None = keep ALL keys from
# the pack; a set() keeps only the listed keys.
_KEEP_KEYS = None

# The 7 base PDM metric keys produced by _pairwise_subscores, saved with _v1 suffix.
_PDM_SCORE_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "pdm_score",
]

# Scoring flags passed to _init_pool (after sim_cfg, scorer_cfg).
_TTC_BOUND = 2.0
_USE_HISTORY_COMFORT = True
_DDC_EXCLUDE_INTERSECTION = True


def _load_scoring_config():
    """Load default PDM scoring config (simulator + scorer) from YAML."""
    from omegaconf import OmegaConf

    config_path = (
        _PROJECT_ROOT / "navsim" / "planning" / "script" / "config"
        / "pdm_scoring" / "default_scoring_parameters.yaml"
    )
    cfg = OmegaConf.load(config_path)
    return cfg.simulator, cfg.scorer


def _compute_prob_density(token, pdm_score):
    """KDE density of the per-token pdm_score distribution, evaluated at each trajectory."""
    try:
        # Small jitter avoids a singular covariance when many scores are identical.
        jittered = pdm_score + np.random.normal(0, 1e-4, size=len(pdm_score))
        return gaussian_kde(jittered)(jittered)
    except Exception as e:
        print(f"Warning: KDE failed for {token}: {e}")
        return np.zeros_like(pdm_score)


def _add_derived_keys(token, reward_dict):
    """Add prob_density and safe_ego_progress_normed to a raw reward-pack entry (if missing)."""
    if "prob_density" not in reward_dict:
        reward_dict["prob_density"] = _compute_prob_density(token, reward_dict["pdm_score"])

    if "safe_ego_progress_normed" not in reward_dict:
        # Trajectories that fail TTC get zero progress. Note: ego_progress itself is
        # also overwritten (it is the scorer's EP target), matching the released cache.
        ttc_fail = reward_dict["time_to_collision_within_bound"] == 0
        safe_ego_progress = np.where(ttc_fail, 0, reward_dict["ego_progress"]).astype(
            reward_dict["ego_progress"].dtype
        )
        reward_dict["ego_progress"] = safe_ego_progress
        max_ep = safe_ego_progress.max()
        reward_dict["safe_ego_progress_normed"] = (
            safe_ego_progress / max_ep if max_ep > 0 else safe_ego_progress
        )
    return reward_dict


def _build_token_list(training_cache_dir):
    """Walk training cache to build list of (token, token_dir, scene_dir)."""
    token_list = []
    training_cache = Path(training_cache_dir)
    for scene_path in sorted(training_cache.iterdir()):
        if not scene_path.is_dir():
            continue
        scene_dir = scene_path.name
        for token_path in sorted(scene_path.iterdir()):
            if not token_path.is_dir():
                continue
            token = token_path.name
            token_list.append((token, str(token_path), scene_dir))
    return token_list


def _process_token(args):
    """Worker function: compute reward + 40-step iPad GT for one token, save to disk."""
    token, token_dir, scene_dir, pdm_cache_dir, reward_entry, traj_8step = args

    metric_cache_path = os.path.join(pdm_cache_dir, scene_dir, "unknown", token, "metric_cache.pkl")

    # Add derived reward keys before simulation, then keep whitelisted keys (None = keep all).
    reward_entry = _add_derived_keys(token, dict(reward_entry))
    if _KEEP_KEYS is None:
        reward_dict = dict(reward_entry)
    else:
        reward_dict = {k: v for k, v in reward_entry.items() if k in _KEEP_KEYS}

    ipad_gt = {}
    if os.path.exists(metric_cache_path):
        try:
            # Run full PDM simulation; subscores has 7 base scores + ego_areas + ttc_time.
            _scores, subscores = _pdm_worker((metric_cache_path, traj_8step))

            # 40-step iPad GT (no downsampling)
            ipad_gt["ego_areas"] = subscores["ego_areas"]  # (8192, 40, 2)
            ipad_gt["ttc_time"] = subscores["ttc_time"]    # (8192, 40)

            # Simulated PDM sub-scores with _v1 suffix
            for key in _PDM_SCORE_KEYS:
                if key in subscores:
                    reward_dict[f"{key}_v1"] = subscores[key]
        except Exception as e:
            print(f"Warning: PDM sim failed for {token} ({scene_dir}): {e}")
            ipad_gt = {}

    # Fill zeros if PDM sim failed or metric cache doesn't exist
    N = traj_8step.shape[0]  # 8192
    if "ego_areas" not in ipad_gt:
        ipad_gt["ego_areas"] = np.zeros((N, 40, 2), dtype=bool)
        ipad_gt["ttc_time"] = np.full((N, 40), _TTC_BOUND, dtype=np.float32)

    reward_dict.update(ipad_gt)

    out_path = os.path.join(token_dir, "transfuser_reward.gz")
    dump_feature_target_to_pickle(Path(out_path), reward_dict)
    return token


def main():
    parser = argparse.ArgumentParser(description="Build per-scene reward + 40-step iPad GT cache")
    parser.add_argument("--training-cache-dir", required=True,
                        help="Path to training cache (has scene_dir/token/ subdirs)")
    parser.add_argument("--pdm-cache-dir", required=True,
                        help="Path to PDM metric cache (has scene_dir/unknown/token/ subdirs)")
    parser.add_argument("--reward-pack-path", required=True,
                        help="Path to the monolithic reward pkl (e.g. GTRS navtrain_8192.pkl)")
    parser.add_argument("--traj-list-path",
                        default="ckpts/8192.npy",
                        help="Path to vocab trajectories .npy (default: ckpts/8192.npy)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tokens with existing transfuser_reward.gz")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Process chunk i of N, format: i/N (e.g. 0/4)")
    args = parser.parse_args()

    # 1. Load vocab trajectories and downsample to 8 steps
    print("Loading vocab trajectories...")
    traj_raw = np.load(args.traj_list_path)  # (8192, 40, 3)
    traj_8step = traj_raw[:, 4::5, :]        # (8192, 8, 3)
    print(f"  Raw shape: {traj_raw.shape}, downsampled: {traj_8step.shape}")

    # 2. Load monolithic reward pack
    print("Loading reward pack...")
    with open(args.reward_pack_path, "rb") as f:
        reward_pack = pickle.load(f)
    print(f"  Loaded {len(reward_pack)} tokens")
    print(f"  Keeping reward-pack keys: {'(all)' if _KEEP_KEYS is None else (sorted(_KEEP_KEYS) or '(none)')}")

    # 3. Build token list from training cache
    print("Building token list...")
    token_list = _build_token_list(args.training_cache_dir)
    print(f"  Found {len(token_list)} tokens in training cache")

    # 4. Filter
    filtered = []
    skipped_no_reward = 0
    skipped_existing = 0
    for token, token_dir, scene_dir in token_list:
        if token not in reward_pack:
            skipped_no_reward += 1
            continue
        if args.skip_existing and os.path.exists(os.path.join(token_dir, "transfuser_reward.gz")):
            skipped_existing += 1
            continue
        filtered.append((token, token_dir, scene_dir))

    print(f"  Skipped {skipped_no_reward} tokens (not in reward pack)")
    print(f"  Skipped {skipped_existing} tokens (already cached)")
    print(f"  Remaining: {len(filtered)} tokens to process")

    # 5. Apply chunking
    if args.chunks:
        chunk_idx, num_chunks = map(int, args.chunks.split("/"))
        chunk_size = (len(filtered) + num_chunks - 1) // num_chunks
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(filtered))
        filtered = filtered[start:end]
        print(f"  Chunk {chunk_idx}/{num_chunks}: processing {len(filtered)} tokens [{start}:{end}]")

    if not filtered:
        print("Nothing to process. Done.")
        return

    # 6. Prepare worker arguments (extract reward entries in main process)
    worker_args = []
    for token, token_dir, scene_dir in filtered:
        reward_entry = reward_pack[token]
        # Convert tensor values to numpy for pickling to workers
        reward_entry_np = {}
        for k, v in reward_entry.items():
            reward_entry_np[k] = v.numpy() if hasattr(v, "numpy") else v
        worker_args.append((token, token_dir, scene_dir, args.pdm_cache_dir, reward_entry_np, traj_8step))

    # 7. Launch pool
    print(f"Launching pool with {args.num_workers} workers...")
    print(f"  ttc_bound={_TTC_BOUND}, use_history_comfort={_USE_HISTORY_COMFORT}, "
          f"ddc_exclude_intersection={_DDC_EXCLUDE_INTERSECTION}")
    sim_cfg, scorer_cfg = _load_scoring_config()

    with mp.Pool(
        args.num_workers,
        initializer=_init_pool,
        initargs=(
            sim_cfg,
            scorer_cfg,
            _TTC_BOUND,
            _USE_HISTORY_COMFORT,
            _DDC_EXCLUDE_INTERSECTION,
        ),
    ) as pool:
        with tqdm(total=len(worker_args), desc="Caching rewards") as pbar:
            for _token in pool.imap_unordered(_process_token, worker_args):
                pbar.update(1)

    print("Done!")


if __name__ == "__main__":
    main()
