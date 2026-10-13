"""
Check integrity of all reward caches (transfuser_reward.gz).

For every token in the training cache, verify the reward dict loads and contains
every expected key: all reward-pack keys, the 7 simulated _v1 scores, the 40-step
iPad GT (ego_areas, ttc_time). The optional ttc_time_b15 from patch_reward.py
is not required.

Usage:
    python scripts/caching/check_reward.py \
        --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
        --num-workers 64
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from multiprocessing import Pool

# Keys copied verbatim from the reward pack.
_PACK_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "pdm_score",
    "prob_density",
    "safe_ego_progress_normed",
]

# 7 simulated scores saved with _v1 suffix by run_train_reward_caching.py.
_V1_KEYS = [
    "no_at_fault_collisions_v1",
    "drivable_area_compliance_v1",
    "driving_direction_compliance_v1",
    "ego_progress_v1",
    "time_to_collision_within_bound_v1",
    "history_comfort_v1",
    "pdm_score_v1",
]

# 40-step iPad GT.
_GT_KEYS = ["ego_areas", "ttc_time"]

_EXPECTED = _PACK_KEYS + _V1_KEYS + _GT_KEYS


def _check(f):
    try:
        with gzip.open(f, "rb") as fh:
            d = pickle.load(fh)
    except Exception as e:
        return (f, f"LOAD_FAIL:{e}")
    missing = [k for k in _EXPECTED if k not in d]
    return (f, missing) if missing else None


def main():
    parser = argparse.ArgumentParser(description="Check reward cache integrity")
    parser.add_argument("--training-cache-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--out", default="/tmp/bad_reward_files.json",
                        help="Where to write the list of bad files")
    args = parser.parse_args()

    print("Building file list...", flush=True)
    files = [str(tp / "transfuser_reward.gz")
             for sp in Path(args.training_cache_dir).iterdir() if sp.is_dir()
             for tp in sp.iterdir()
             if (tp / "transfuser_reward.gz").exists()]
    print(f"  Found {len(files)} reward files", flush=True)
    print(f"  Expecting {len(_EXPECTED)} keys per file", flush=True)

    bad = []
    with Pool(args.num_workers) as p:
        for i, r in enumerate(p.imap_unordered(_check, files, chunksize=64)):
            if r:
                bad.append(r)
            if (i + 1) % 10000 == 0:
                print(f"  checked {i+1}/{len(files)}, bad so far {len(bad)}", flush=True)

    print(f"\nBad files: {len(bad)} / {len(files)}")
    for f, why in bad[:50]:
        print(f"  {why if isinstance(why, str) else 'missing ' + str(why)}: {f}")
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more")

    json.dump([f for f, _ in bad], open(args.out, "w"))
    print(f"\nWrote {len(bad)} bad paths to {args.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
