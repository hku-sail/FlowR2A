"""
Analyze scene clip structure in a navsim split.

Scans all log .pkl files, groups frames by log_name, sorts by timestamp,
and identifies continuous clips where adjacent frames are <= 0.55s apart.

Tokens can be restricted to a split's scene filter (e.g. navtest's 12k tokens)
or to the tokens in an eval full_results.pkl. The saved index is what the
inspectors read via their clip index (`--clip-index` / DEFAULT_CLIP_INDEX).

Usage:
    # All raw frames in test split
    python scripts/visualization/analyze_scene_clips.py --split test

    # Only navtest tokens (from the navtest scene filter)
    python scripts/visualization/analyze_scene_clips.py --split test \
        --scene-filter navtest \
        --save scripts/visualization/navtest_clips_index.pkl

    # Only tokens that were evaluated (full_results.pkl from run_eval.py --save_full_results)
    python scripts/visualization/analyze_scene_clips.py --split test \
        --eval-pkl /path/to/full_results.pkl
"""

import argparse
import pickle
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from tqdm import tqdm

SCENE_FILTER_DIR = Path(__file__).resolve().parents[2] / \
    "navsim/planning/script/config/common/train_test_split/scene_filter"


def analyze_clips(
    data_path: Path,
    eval_tokens: Optional[Set[str]] = None,
    log_names: Optional[Set[str]] = None,
    time_gap_threshold_us: float = 550_000,  # 0.55s in microseconds
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Scan all logs and find continuous scene clips.

    Args:
        data_path: path to navsim_logs/<split>/
        eval_tokens: if provided, only include these tokens in clips
        log_names: if provided, only scan these logs
        time_gap_threshold_us: max gap between adjacent frames in microseconds

    Returns:
        all_clips: list of clip dicts
        log_to_clips: mapping from log_name to its clips
    """
    # Collect all frames grouped by log_name
    log_frames: Dict[str, List[dict]] = defaultdict(list)

    log_files = sorted(data_path.glob("*.pkl"))
    if log_names is not None:
        log_files = [p for p in log_files if p.stem in log_names]
    print(f"Scanning {len(log_files)} log files in {data_path}")

    for log_path in tqdm(log_files, desc="Loading logs"):
        with open(log_path, "rb") as f:
            frames = pickle.load(f)
        for frame in frames:
            token = frame["token"]
            if eval_tokens is not None and token not in eval_tokens:
                continue
            log_frames[frame["log_name"]].append({
                "token": token,
                "timestamp": frame["timestamp"],
            })

    # Sort each log's frames by timestamp and find clips
    all_clips = []
    log_to_clips = defaultdict(list)

    for log_name in sorted(log_frames.keys()):
        frames = sorted(log_frames[log_name], key=lambda f: f["timestamp"])

        clip_tokens = [frames[0]["token"]]
        clip_timestamps = [frames[0]["timestamp"]]

        for i in range(1, len(frames)):
            time_diff = frames[i]["timestamp"] - frames[i - 1]["timestamp"]

            if time_diff <= time_gap_threshold_us:
                clip_tokens.append(frames[i]["token"])
                clip_timestamps.append(frames[i]["timestamp"])
            else:
                # Save current clip
                clip = {
                    "log_name": log_name,
                    "tokens": clip_tokens,
                    "timestamps": clip_timestamps,
                    "num_frames": len(clip_tokens),
                    "duration_s": (clip_timestamps[-1] - clip_timestamps[0]) / 1e6,
                }
                all_clips.append(clip)
                log_to_clips[log_name].append(clip)

                # Start new clip
                clip_tokens = [frames[i]["token"]]
                clip_timestamps = [frames[i]["timestamp"]]

        # Don't forget the last clip
        clip = {
            "log_name": log_name,
            "tokens": clip_tokens,
            "timestamps": clip_timestamps,
            "num_frames": len(clip_tokens),
            "duration_s": (clip_timestamps[-1] - clip_timestamps[0]) / 1e6,
        }
        all_clips.append(clip)
        log_to_clips[log_name].append(clip)

    return all_clips, log_to_clips

def main():
    parser = argparse.ArgumentParser(description="Analyze scene clip structure")
    parser.add_argument("--split", default="test", help="Data split (test/trainval)")
    parser.add_argument("--scene-filter", default=None,
                        help="Scene filter name under train_test_split/scene_filter (e.g. navtest, navtrain) "
                             "to restrict to its logs/tokens")
    parser.add_argument("--eval-pkl", default=None, help="full_results.pkl to filter tokens to eval set only")
    parser.add_argument("--save", default=None, help="Save clip index as pkl (optional)")
    args = parser.parse_args()

    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_path = openscene_root / "navsim_logs" / args.split

    # Load scene filter tokens/logs if provided
    eval_tokens, log_names = None, None
    if args.scene_filter:
        filter_path = SCENE_FILTER_DIR / f"{args.scene_filter}.yaml"
        print(f"Loading scene filter from {filter_path}")
        with open(filter_path) as f:
            scene_filter = yaml.safe_load(f)
        if scene_filter.get("log_names"):
            log_names = set(scene_filter["log_names"])
        if scene_filter.get("tokens"):
            eval_tokens = set(scene_filter["tokens"])
        print(f"Scene filter logs: {len(log_names or [])}, tokens: {len(eval_tokens or [])}")

    # Load eval tokens if provided (intersected with the scene filter tokens)
    if args.eval_pkl:
        print(f"Loading eval tokens from {args.eval_pkl}")
        with open(args.eval_pkl, "rb") as f:
            results = pickle.load(f)
        pkl_tokens = {r["token"] for r in results}
        eval_tokens = pkl_tokens if eval_tokens is None else eval_tokens & pkl_tokens
        print(f"Eval token count: {len(eval_tokens)}")

    all_clips, log_to_clips = analyze_clips(data_path, eval_tokens=eval_tokens, log_names=log_names)

    # Statistics
    num_clips = len(all_clips)
    num_logs = len(log_to_clips)
    clip_lengths = np.array([c["num_frames"] for c in all_clips])
    clip_durations = np.array([c["duration_s"] for c in all_clips])
    total_frames = clip_lengths.sum()

    print(f"\n{'='*60}")
    print(f"Split: {args.split}")
    if args.scene_filter:
        print(f"Scene filter: {args.scene_filter}")
    if eval_tokens:
        print(f"Filtered to {len(eval_tokens)} tokens")
    print(f"{'='*60}")
    print(f"Total logs:           {num_logs}")
    print(f"Total frames:         {total_frames}")
    print(f"Total clips:          {num_clips}")
    print(f"")
    print(f"Clip length (frames):")
    print(f"  min:    {clip_lengths.min()}")
    print(f"  max:    {clip_lengths.max()}")
    print(f"  mean:   {clip_lengths.mean():.1f}")
    print(f"  median: {np.median(clip_lengths):.0f}")
    print(f"")
    print(f"Clip duration (seconds):")
    print(f"  min:    {clip_durations.min():.1f}s")
    print(f"  max:    {clip_durations.max():.1f}s")
    print(f"  mean:   {clip_durations.mean():.1f}s")
    print(f"  median: {np.median(clip_durations):.1f}s")
    print(f"")

    # Distribution
    print(f"Clip length distribution:")
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    for i in range(len(bins) - 1):
        count = np.sum((clip_lengths >= bins[i]) & (clip_lengths < bins[i + 1]))
        if count > 0:
            print(f"  [{bins[i]:5d}, {bins[i+1]:5d}):  {count:5d} clips")
    count = np.sum(clip_lengths >= bins[-1])
    if count > 0:
        print(f"  [{bins[-1]:5d},     +):  {count:5d} clips")

    print(f"\nClips per log:")
    clips_per_log = [len(v) for v in log_to_clips.values()]
    print(f"  min: {min(clips_per_log)}, max: {max(clips_per_log)}, mean: {np.mean(clips_per_log):.1f}")

    # Show longest clips
    print(f"\nTop 10 longest clips:")
    sorted_clips = sorted(all_clips, key=lambda c: c["num_frames"], reverse=True)
    for i, c in enumerate(sorted_clips[:10]):
        print(f"  {i+1}. {c['log_name']}  {c['num_frames']} frames  {c['duration_s']:.1f}s")

    # Save if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"clips": all_clips, "log_to_clips": dict(log_to_clips)}, f)
        print(f"\nSaved clip index to {save_path}")


if __name__ == "__main__":
    main()
