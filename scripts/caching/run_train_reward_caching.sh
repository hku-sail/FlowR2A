#!/bin/bash

# Usage: bash scripts/caching/run_train_reward_caching.sh [CHUNK_ID] [TOTAL_CHUNKS]
# Example: bash scripts/caching/run_train_reward_caching.sh 0
#          bash scripts/caching/run_train_reward_caching.sh 1 4

CHUNK_ID=${1:-0}
TOTAL_CHUNKS=${2:-4}

python scripts/caching/run_train_reward_caching.py \
     --training-cache-dir $NAVSIM_EXP_ROOT/training_cache \
     --pdm-cache-dir $NAVSIM_EXP_ROOT/train_pdm_cache \
     --reward-pack-path ckpts/navtrain_8192.pkl \
     --num-workers 64 \
     --skip-existing \
     --chunks ${CHUNK_ID}/${TOTAL_CHUNKS}
