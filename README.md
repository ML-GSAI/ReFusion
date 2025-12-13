# ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)]()
[![deploy](https://img.shields.io/badge/Hugging%20Face-ReFusion-FFEB3B)](https://huggingface.co/GSAI-ML/ReFusion)

## Introduction
We introduce ReFusion, a novel masked diffusion model featuring two core innovations:
1. It unifies a causal attention mechanism with global, any-order slot generation, **enabling full KV cache reuse without sacrificing flexibility**.
2. It simplifies the learning objective from an intractable token-combination space to a manageable slot-permutation space, **significantly boosting learning efficiency**.

Empirically, ReFusion not only outperforms prior MDMs with a **34% performance gain** and an over **18× speedup** on average, but also bridges the performance gap to strong ARMs while maintaining a **2.33× average speedup**.

<div align="center">
  <img src="./images/models.png" width="50%" alt="ReFusion's performance" />

  <br>

  <em>Figure: ReFusion achieves the best balance of speed and accuracy on MBPP. Metrics are calculated relative to the Qwen3-8B baseline.</em>
</div>

## Usage
### Environment Setup
```bash
git clone https://github.com/ML-GSAI/ReFusion.git
cd ReFusion
conda env create -f refusion_full_env.yml
conda activate refusion_py10
```

### Training
