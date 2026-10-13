<div align="center">
<h1>FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning</h1>

<h3>NeurIPS 2026</h3>

<a href="https://lixirui142.github.io/flowr2a-ad/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2606.24231"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/lixirui142/FlowR2A"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow.svg" alt="Hugging Face Model"></a>
[![Papers with Code: #2 on NAVSIM v1](https://paperswithcode.co/api/v1/papers/2606.24231/leaderboard-badge.svg?eval=23717&live=1)](https://paperswithcode.co/api/v1/papers/2606.24231/leaderboard-badge-link?eval=23717)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

## News
- **[2026-10]** FlowR2A has been accepted to NeurIPS 2026.

## Overview
FlowR2A is a multimodal driving planner that learns the reward-conditioned action distribution *p(a | r)* with flow matching. Instead of treating simulation rewards as discriminative targets, FlowR2A treats them as a *condition*, unifying the dense supervision of scoring-based methods with the generative proposal modeling of anchor-based methods. At inference, generation is steered toward high-reward trajectories via classifier-free guidance.

<div align="center">
<img src="assets/pipeline.png" width="100%"/>
<p><b>Training pipeline of FlowR2A.</b></p>
</div>


## Table of Contents
- [News](#news)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Qualitative Results on NAVSIM](#qualitative-results-on-navsim)
- [Getting Started](#getting-started)
- [To-Do](#to-do)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)


## Qualitative Results on NAVSIM
Each row is one planner; trajectory proposals are colored by PDM score, from <b>red&nbsp;(0)</b> to <b>green&nbsp;(1)</b>. FlowR2A (bottom row) produces proposals that are both diverse and consistently high-scoring.

<div align="center">
  <img src="assets/qualitative_134.gif" width="49%"/>
  <img src="assets/qualitative_240.gif" width="49%"/>
</div>

> More qualitative comparisons are available on the [project page](https://lixirui142.github.io/flowr2a-ad/).

## Getting Started

- [Data and Environment Preparation](docs/install.md) — install dependencies and download the data/models needed for evaluation.
- [Evaluation](docs/evaluation.md) — download the pre-trained checkpoints and run evaluation on the test set.
- [Training](docs/training.md) — download the training data and prepare the cache, then train the model in two stages.
- [Reward Simulation](docs/reward_simulation.md) — (optional) generate the reward cache from scratch instead of downloading it.

## To-Do
- [x] Inference and evaluation code
- [x] Training code
- [x] Reward simulation / caching pipeline

## Acknowledgement
FlowR2A is built upon the following outstanding open-source contributions: [NAVSIM](https://github.com/autonomousvision/navsim), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive).

## Citation
If you find FlowR2A useful in your research or applications, please consider giving us a star 🌟 and citing it with the following BibTeX entry.

```bibtex
@inproceedings{flowr2a2026,
  title     = {FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning},
  author    = {Li, Xirui and Liu, Zhe and Ye, Xiaoqing and Han, Wenhua and Pan, Yifeng and Han, Junyu and Zhao, Hengshuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License
MIT — see [LICENSE](LICENSE).
